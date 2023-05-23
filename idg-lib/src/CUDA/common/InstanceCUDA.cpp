// Copyright (C) 2023 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <iomanip>  // setprecision

#include "InstanceCUDA.h"
#include "PowerRecord.h"
#include "kernels/KernelGridder.cuh"
#include "kernels/KernelDegridder.cuh"

using namespace idg::kernel;

/*
 * Option to enable repeated kernel invocations
 * this is used to measure energy consumpton
 * using a low-resolution power measurement (NVML)
 */
#define ENABLE_REPEAT_KERNELS 0
#define NR_REPETITIONS_GRIDDER 10
#define NR_REPETITIONS_ADDER 50
#define NR_REPETITIONS_GRID_FFT 500

namespace idg::kernel::cuda {

// Constructor
InstanceCUDA::InstanceCUDA(ProxyInfo& info, size_t device_id)
    : KernelsInstance(), proxy_info_(info) {
#if defined(DEBUG)
  std::cout << __func__ << std::endl;
#endif

  // Initialize members
  device_.reset(new cu::Device(device_id));
  context_.reset(new cu::Context(*device_));
  profiler_.reset(new cu::Profiler(*context_));
  stream_execute_.reset(new cu::Stream(*context_));
  stream_htod_.reset(new cu::Stream(*context_));
  stream_dtoh_.reset(new cu::Stream(*context_));
  power_meter_ = pmt::get_power_meter(pmt::sensor_device, device_id);

  // Compile kernels
  compile_kernels();

  // Load kernels
  load_kernels();
}

// Destructor
InstanceCUDA::~InstanceCUDA() {
#if defined(DEBUG)
  std::cout << __func__ << std::endl;
#endif

  free_subgrid_fft();
  free_events();
  modules_.clear();
  stream_execute_.reset();
  stream_htod_.reset();
  stream_dtoh_.reset();
  profiler_.reset();
  context_.reset();
  device_.reset();
}

/*
    Compilation
*/
std::string InstanceCUDA::get_compiler_flags() {
  // Constants
  std::stringstream flags_constants;

  // CUDA specific flags
  std::stringstream flags_cuda;
  flags_cuda << "-use_fast_math";
#if defined(CUDA_KERNEL_DEBUG)
  flags_cuda << " -G ";
#else
  flags_cuda << " -lineinfo";
#endif
  flags_cuda << " -src-in-ptx";
#if defined(USE_EXTRAPOLATE)
  flags_cuda << " -DUSE_EXTRAPOLATE";
#endif

  // Device specific flags
  int capability = (*device_).get_capability();
  std::stringstream flags_device;
  flags_device << "-arch=sm_" << capability;

  // Include flags
  std::stringstream flags_includes;
  flags_includes << "-I" << auxiliary::get_inc_dir();

  // Combine flags
  std::string flags = " " + flags_cuda.str() + " " + flags_device.str() + " " +
                      flags_constants.str() + " " + flags_includes.str();
  return flags;
}

cu::Module* InstanceCUDA::compile_kernel(std::string& flags, std::string& src,
                                         std::string& bin) {
  // Create a string with the full path to the cubin file "kernel.cubin"
  std::string lib = proxy_info_.get_path_to_lib() + "/" + bin;

  // Create a string for all sources that are combined
  std::string source = proxy_info_.get_path_to_src() + "/" + src;

  // Call the compiler
  cu::Source(source.c_str()).compile(lib.c_str(), flags.c_str());

  // Create module
  return new cu::Module(*context_, lib.c_str());
}

void InstanceCUDA::compile_kernels() {
#if defined(DEBUG)
  std::cout << __func__ << std::endl;
#endif

  // Get source directory
  std::string srcdir = auxiliary::get_lib_dir() + "/idg-cuda";
#if defined(DEBUG)
  std::cout << "Searching for source files in: " << srcdir << std::endl;
#endif

  // Create temp directory
  char tmpdir[] = "/tmp/idg-XXXXXX";
  char* tmpdir_ = mkdtemp(tmpdir);
  if (!tmpdir_) {
    throw std::runtime_error("could not create tmp directory.");
  }
#if defined(DEBUG)
  std::cout << "Temporary files will be stored in: " << tmpdir << std::endl;
#endif

  // Get compiler flags
  std::string flags_common = get_compiler_flags();

  // Create vector of source filenames, filenames and flags
  std::vector<std::string> src;
  std::vector<std::string> cubin;
  std::vector<std::string> flags;

  // Gridder
  src.push_back("KernelGridder.cu");
  cubin.push_back("Gridder.cubin");
  flags.push_back(flags_common);

  // Degridder
  src.push_back("KernelDegridder.cu");
  cubin.push_back("Degridder.cubin");
  flags.push_back(flags_common);

  // Scaler
  src.push_back("KernelScaler.cu");
  cubin.push_back("Scaler.cubin");
  flags.push_back(flags_common);

  // Adder
  src.push_back("KernelAdder.cu");
  cubin.push_back("Adder.cubin");
  std::stringstream flags_adder;
  flags_adder << flags_common;
  flags_adder << " -DTILE_SIZE_GRID=" << kTileSizeGrid;
  flags.push_back(flags_adder.str());

  // Splitter
  src.push_back("KernelSplitter.cu");
  cubin.push_back("Splitter.cubin");
  std::stringstream flags_splitter;
  flags_splitter << flags_common;
  flags_splitter << " -DTILE_SIZE_GRID=" << kTileSizeGrid;
  flags.push_back(flags_splitter.str());

  // Calibrate
  src.push_back("KernelCalibrate.cu");
  cubin.push_back("Calibrate.cubin");
  flags.push_back(flags_common);

  // Average beam
  src.push_back("KernelAverageBeam.cu");
  cubin.push_back("AverageBeam.cubin");
  flags.push_back(flags_common);

  // FFT shift
  src.push_back("KernelFFTShift.cu");
  cubin.push_back("KernelFFTShift.cubin");
  flags.push_back(flags_common);

  // W-Tiling
  src.push_back("KernelWtiling.cu");
  cubin.push_back("KernelWtiling.cubin");
  flags.push_back(flags_common);

  // Compile all kernels
  for (unsigned i = 0; i < src.size(); i++) {
    modules_.push_back(std::unique_ptr<cu::Module>());
  }
#pragma omp parallel for
  for (unsigned i = 0; i < src.size(); i++) {
    modules_[i].reset(compile_kernel(flags[i], src[i], cubin[i]));
  }
}

void InstanceCUDA::load_kernels() {
  CUfunction function;
  unsigned found = 0;

  // Load gridder function
  if (cuModuleGetFunction(&function, *modules_[0], kNameGridder.c_str()) ==
      CUDA_SUCCESS) {
    function_gridder_.reset(new cu::Function(*context_, function));
    found++;
  }

  // Load degridder function
  if (cuModuleGetFunction(&function, *modules_[1], kNameDegridder.c_str()) ==
      CUDA_SUCCESS) {
    function_degridder_.reset(new cu::Function(*context_, function));
    found++;
  }

  // Load scalar function
  if (cuModuleGetFunction(&function, *modules_[2], kNameScaler.c_str()) ==
      CUDA_SUCCESS) {
    function_scaler_.reset(new cu::Function(*context_, function));
    found++;
  }

  // Load adder function
  if (cuModuleGetFunction(&function, *modules_[3], kNameAdder.c_str()) ==
      CUDA_SUCCESS) {
    function_adder_.reset(new cu::Function(*context_, function));
    found++;
  }

  // Load splitter function
  if (cuModuleGetFunction(&function, *modules_[4], kNameSplitter.c_str()) ==
      CUDA_SUCCESS) {
    function_splitter_.reset(new cu::Function(*context_, function));
    found++;
  }

  // Load calibration functions
  if (cuModuleGetFunction(&function, *modules_[5],
                          kNnameCalibrateLMNP.c_str()) == CUDA_SUCCESS) {
    functions_calibrate_.emplace_back(new cu::Function(*context_, function));
    found++;
  }
  if (cuModuleGetFunction(&function, *modules_[5],
                          kNameCalibrateSums.c_str()) == CUDA_SUCCESS) {
    functions_calibrate_.emplace_back(new cu::Function(*context_, function));
  }
  if (cuModuleGetFunction(&function, *modules_[5],
                          kNameCalibrateGradient.c_str()) == CUDA_SUCCESS) {
    functions_calibrate_.emplace_back(new cu::Function(*context_, function));
  }
  if (cuModuleGetFunction(&function, *modules_[5],
                          kNameCalibrateHessian.c_str()) == CUDA_SUCCESS) {
    functions_calibrate_.emplace_back(new cu::Function(*context_, function));
  }

  // Load average beam function
  if (cuModuleGetFunction(&function, *modules_[6], kNameAverageBeam.c_str()) ==
      CUDA_SUCCESS) {
    function_average_beam_.reset(new cu::Function(*context_, function));
    found++;
  }

  // Load FFT shift function
  if (cuModuleGetFunction(&function, *modules_[7], kNameFftShift.c_str()) ==
      CUDA_SUCCESS) {
    function_fft_shift_.reset(new cu::Function(*context_, function));
    found++;
  }

  // Load W-Tiling functions
  if (cuModuleGetFunction(&function, *modules_[8], kNameCopyTiles.c_str()) ==
      CUDA_SUCCESS) {
    functions_wtiling_.emplace_back(new cu::Function(*context_, function));
    found++;
  }
  if (cuModuleGetFunction(&function, *modules_[8], kNameApplyPhasor.c_str()) ==
      CUDA_SUCCESS) {
    functions_wtiling_.emplace_back(new cu::Function(*context_, function));
  }
  if (cuModuleGetFunction(&function, *modules_[8],
                          kNameSubgridsToWtiles.c_str()) == CUDA_SUCCESS) {
    functions_wtiling_.emplace_back(new cu::Function(*context_, function));
  }
  if (cuModuleGetFunction(&function, *modules_[8], kNameWtilesToGrid.c_str()) ==
      CUDA_SUCCESS) {
    functions_wtiling_.emplace_back(new cu::Function(*context_, function));
  }
  if (cuModuleGetFunction(&function, *modules_[8],
                          kNameSubgridsFromWtiles.c_str()) == CUDA_SUCCESS) {
    functions_wtiling_.emplace_back(new cu::Function(*context_, function));
  }
  if (cuModuleGetFunction(&function, *modules_[8],
                          kNameWtilesFromGrid.c_str()) == CUDA_SUCCESS) {
    functions_wtiling_.emplace_back(new cu::Function(*context_, function));
  }
  if (cuModuleGetFunction(&function, *modules_[8],
                          kNameWtilesToPatch.c_str()) == CUDA_SUCCESS) {
    functions_wtiling_.emplace_back(new cu::Function(*context_, function));
  }
  if (cuModuleGetFunction(&function, *modules_[8],
                          kNameWtilesFromPatch.c_str()) == CUDA_SUCCESS) {
    functions_wtiling_.emplace_back(new cu::Function(*context_, function));
  }

  // Verify that all functions are found
  if (found != modules_.size()) {
    std::cerr << "Incorrect number of functions found: " << found
              << " != " << modules_.size() << std::endl;
    exit(EXIT_FAILURE);
  }
}

std::ostream& operator<<(std::ostream& os, InstanceCUDA& d) {
  cu::ScopedContext scc(d.get_context());

  os << d.get_device().get_name() << std::endl;
  os << std::setprecision(2);
  os << std::fixed;

  // Device memory
  auto device_memory_total =
      d.get_device().get_total_memory() / (1024 * 1024);  // MBytes
  auto device_memory_free =
      d.get_device().get_free_memory() / (1024 * 1024);  // MBytes
  os << "\tDevice memory : " << device_memory_free << " Mb  / "
     << device_memory_total << " Mb (free / total)" << std::endl;

  // Shared memory
  auto shared_memory =
      d.get_device()
          .get_attribute<
              CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK>();  // Bytes
  os << "\tShared memory : " << shared_memory / (float)1024 << " Kb"
     << std::endl;

  // Frequencies
  auto clock_frequency =
      d.get_device().get_attribute<CU_DEVICE_ATTRIBUTE_CLOCK_RATE>() /
      1000;  // Mhz
  os << "\tClk frequency : " << clock_frequency << " Ghz" << std::endl;
  auto mem_frequency =
      d.get_device().get_attribute<CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE>() /
      1000;  // Mhz
  os << "\tMem frequency : " << mem_frequency << " Ghz" << std::endl;

  // Cores/bus
  auto nr_sm =
      d.get_device().get_attribute<CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT>();
  auto mem_bus_width =
      d.get_device()
          .get_attribute<
              CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH>();  // Bits
  os << "\tNumber of SM  : " << nr_sm << std::endl;
  os << "\tMem bus width : " << mem_bus_width << " bit" << std::endl;
  os << "\tMem bandwidth : " << 2 * (mem_bus_width / 8) * mem_frequency / 1000
     << " GB/s" << std::endl;

  auto nr_threads =
      d.get_device()
          .get_attribute<CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR>();
  os << "\tNumber of threads  : " << nr_threads << std::endl;

  // Misc
  os << "\tCapability    : " << d.get_device().get_capability() << std::endl;

  // Unified memory
  auto supports_managed_memory =
      d.get_device().get_attribute<CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY>();
  os << "\tUnified memory : " << supports_managed_memory << std::endl;

  os << std::endl;
  return os;
}

pmt::State InstanceCUDA::measure() { return power_meter_->Read(); }

void InstanceCUDA::measure(PowerRecord& record, cu::Stream& stream) {
  record.sensor = *power_meter_;
  record.enqueue(stream);
}

cu::Event& InstanceCUDA::get_event() {
  // Create new event
  cu::Event* event = new cu::Event(*context_);

  // This event is used in a callback, where it can not be destroyed
  // after use. Instead, register the event globally, and take care of
  // destruction there.
  events.emplace_back(event);

  // Return a reference to the event
  return *event;
}

typedef struct {
  PowerRecord* start;
  PowerRecord* end;
  std::shared_ptr<Report> report;
  Report::ID id;
  void (Report::*update_report)(pmt::State&, pmt::State&);
} UpdateData;

UpdateData* get_update_data(cu::Event& event, pmt::Pmt& sensor,
                            std::shared_ptr<Report> report, Report::ID id) {
  UpdateData* data = new UpdateData();
  data->start = new PowerRecord(event, sensor);
  data->end = new PowerRecord(event, sensor);
  data->report = report;
  data->id = id;
  return data;
}

void update_report_callback(CUstream, CUresult, void* userData) {
  UpdateData* data = static_cast<UpdateData*>(userData);
  PowerRecord* start = data->start;
  PowerRecord* end = data->end;
  data->report->update(data->id, start->state, end->state);
  delete start;
  delete end;
  delete data;
}

void InstanceCUDA::start_measurement(void* ptr) {
  UpdateData* data = (UpdateData*)ptr;

  // Schedule the first measurement (prior to kernel execution)
  data->start->enqueue(*stream_execute_);
}

void InstanceCUDA::end_measurement(void* ptr) {
  UpdateData* data = (UpdateData*)ptr;

  // Schedule the second measurement (after the kernel execution)
  data->end->enqueue(*stream_execute_);

  // Afterwards, update the report according to the two measurements
  stream_execute_->addCallback(
      static_cast<CUstreamCallback>(&update_report_callback), data);
}

void InstanceCUDA::launch_gridder(
    int time_offset, int nr_subgrids, int nr_polarizations, int grid_size,
    int subgrid_size, float image_size, float w_step, int nr_channels,
    int nr_stations, float shift_l, float shift_m, cu::DeviceMemory& d_uvw,
    cu::DeviceMemory& d_wavenumbers, cu::DeviceMemory& d_visibilities,
    cu::DeviceMemory& d_taper, cu::DeviceMemory& d_aterms,
    cu::DeviceMemory& d_aterm_indices, cu::DeviceMemory& d_metadata,
    cu::DeviceMemory& d_avg_aterm, cu::DeviceMemory& d_subgrid) {
  const void* parameters[] = {&time_offset,
                              &nr_polarizations,
                              &grid_size,
                              &subgrid_size,
                              &image_size,
                              &w_step,
                              &shift_l,
                              &shift_m,
                              &nr_channels,
                              &nr_stations,
                              d_uvw.data(),
                              d_wavenumbers.data(),
                              d_visibilities.data(),
                              d_taper.data(),
                              d_aterms.data(),
                              d_aterm_indices.data(),
                              d_metadata.data(),
                              d_avg_aterm.data(),
                              d_subgrid.data()};

  dim3 grid(nr_subgrids);
  dim3 block(KernelGridder::block_size_x);
  UpdateData* data =
      get_update_data(get_event(), *power_meter_, report_, Report::gridder);
  start_measurement(data);
#if ENABLE_REPEAT_KERNELS
  for (int i = 0; i < NR_REPETITIONS_GRIDDER; i++)
#endif
    stream_execute_->launchKernel(*function_gridder_, grid, block, 0,
                                  parameters);
  end_measurement(data);
}

void InstanceCUDA::launch_degridder(
    int time_offset, int nr_subgrids, int nr_polarizations, int grid_size,
    int subgrid_size, float image_size, float w_step, int nr_channels,
    int nr_stations, float shift_l, float shift_m, cu::DeviceMemory& d_uvw,
    cu::DeviceMemory& d_wavenumbers, cu::DeviceMemory& d_visibilities,
    cu::DeviceMemory& d_taper, cu::DeviceMemory& d_aterms,
    cu::DeviceMemory& d_aterm_indices, cu::DeviceMemory& d_metadata,
    cu::DeviceMemory& d_subgrid) {
  const void* parameters[] = {&time_offset,
                              &nr_polarizations,
                              &grid_size,
                              &subgrid_size,
                              &image_size,
                              &w_step,
                              &shift_l,
                              &shift_m,
                              &nr_channels,
                              &nr_stations,
                              d_uvw.data(),
                              d_wavenumbers.data(),
                              d_visibilities.data(),
                              d_taper.data(),
                              d_aterms.data(),
                              d_aterm_indices.data(),
                              d_metadata.data(),
                              d_subgrid.data()};

  dim3 grid(nr_subgrids);
  dim3 block(KernelDegridder::block_size_x);

  UpdateData* data =
      get_update_data(get_event(), *power_meter_, report_, Report::degridder);
  start_measurement(data);
#if ENABLE_REPEAT_KERNELS
  for (int i = 0; i < NR_REPETITIONS_GRIDDER; i++)
#endif
    stream_execute_->launchKernel(*function_degridder_, grid, block, 0,
                                  parameters);
  end_measurement(data);
}

void InstanceCUDA::launch_average_beam(
    int nr_baselines, int nr_antennas, int nr_timesteps, int nr_channels,
    int nr_aterms, int subgrid_size, cu::DeviceMemory& d_uvw,
    cu::DeviceMemory& d_baselines, cu::DeviceMemory& d_aterms,
    cu::DeviceMemory& d_aterm_offsets, cu::DeviceMemory& d_weights,
    cu::DeviceMemory& d_average_beam) {
  const void* parameters[] = {
      &nr_antennas,       &nr_timesteps,        &nr_channels,
      &nr_aterms,         &subgrid_size,        d_uvw.data(),
      d_baselines.data(), d_aterms.data(),      d_aterm_offsets.data(),
      d_weights.data(),   d_average_beam.data()};

  dim3 grid(nr_baselines);
  dim3 block(128);

  UpdateData* data = get_update_data(get_event(), *power_meter_, report_,
                                     Report::average_beam);
  start_measurement(data);
  stream_execute_->launchKernel(*function_average_beam_, grid, block, 0,
                                parameters);
  end_measurement(data);
}

void InstanceCUDA::launch_calibrate(
    int nr_subgrids, int grid_size, int subgrid_size, float image_size,
    float w_step, int total_nr_timesteps, int nr_channels, int nr_stations,
    int nr_terms, cu::DeviceMemory& d_uvw, cu::DeviceMemory& d_wavenumbers,
    cu::DeviceMemory& d_visibilities, cu::DeviceMemory& d_weights,
    cu::DeviceMemory& d_aterm, cu::DeviceMemory& d_aterm_derivatives,
    cu::DeviceMemory& d_aterm_indices, cu::DeviceMemory& d_metadata,
    cu::DeviceMemory& d_subgrid, cu::DeviceMemory& d_sums1,
    cu::DeviceMemory& d_sums2, cu::DeviceMemory& d_lmnp,
    cu::DeviceMemory& d_hessian, cu::DeviceMemory& d_gradient,
    cu::DeviceMemory& d_residual) {
  dim3 grid(nr_subgrids);
  dim3 block(128);
  UpdateData* data =
      get_update_data(get_event(), *power_meter_, report_, Report::calibrate);
  start_measurement(data);

  // Get functions
  std::unique_ptr<cu::Function>& function_lmnp = functions_calibrate_[0];
  std::unique_ptr<cu::Function>& function_sums = functions_calibrate_[1];
  std::unique_ptr<cu::Function>& function_gradient = functions_calibrate_[2];
  std::unique_ptr<cu::Function>& function_hessian = functions_calibrate_[3];

  // Precompute l,m,n and phase offset
  const void* parameters_lmnp[] = {&grid_size,        &subgrid_size,
                                   &image_size,       &w_step,
                                   d_metadata.data(), d_lmnp.data()};
  stream_execute_->launchKernel(*function_lmnp, grid, block, 0,
                                parameters_lmnp);

  const unsigned int nr_polarizations = 4;
  unsigned int max_nr_terms = 8;
  unsigned int current_nr_terms_y = max_nr_terms;
  for (unsigned int term_offset_y = 0; term_offset_y < (unsigned int)nr_terms;
       term_offset_y += current_nr_terms_y) {
    unsigned int last_term_y =
        min(nr_terms, term_offset_y + current_nr_terms_y);
    unsigned int current_nr_terms_y = last_term_y - term_offset_y;

    // Compute sums1
    const void* parameters_sums[] = {&nr_polarizations,
                                     &subgrid_size,
                                     &image_size,
                                     &total_nr_timesteps,
                                     &nr_channels,
                                     &nr_stations,
                                     &term_offset_y,
                                     &current_nr_terms_y,
                                     &nr_terms,
                                     d_uvw.data(),
                                     d_wavenumbers.data(),
                                     d_aterm.data(),
                                     d_aterm_derivatives.data(),
                                     d_aterm_indices.data(),
                                     d_metadata.data(),
                                     d_subgrid.data(),
                                     d_sums1.data(),
                                     d_lmnp.data()};
    stream_execute_->launchKernel(*function_sums, grid, block, 0,
                                  parameters_sums);

    // Compute gradient (diagonal)
    if (term_offset_y == 0) {
      const void* parameters_gradient[] = {&nr_polarizations,
                                           &subgrid_size,
                                           &image_size,
                                           &total_nr_timesteps,
                                           &nr_channels,
                                           &nr_stations,
                                           &term_offset_y,
                                           &current_nr_terms_y,
                                           &nr_terms,
                                           d_uvw.data(),
                                           d_wavenumbers.data(),
                                           d_visibilities.data(),
                                           d_weights.data(),
                                           d_aterm.data(),
                                           d_aterm_derivatives.data(),
                                           d_aterm_indices.data(),
                                           d_metadata.data(),
                                           d_subgrid.data(),
                                           d_sums1.data(),
                                           d_lmnp.data(),
                                           d_gradient.data(),
                                           d_residual.data()};
      stream_execute_->launchKernel(*function_gradient, grid, block, 0,
                                    parameters_gradient);
    }

    // Compute hessian (diagonal)
    const void* parameters_hessian1[] = {
        &nr_polarizations, &total_nr_timesteps,    &nr_channels,
        &term_offset_y,    &term_offset_y,         &nr_terms,
        d_weights.data(),  d_aterm_indices.data(), d_metadata.data(),
        d_sums1.data(),    d_sums1.data(),         d_hessian.data()};
    dim3 block_hessian(current_nr_terms_y, current_nr_terms_y);
    stream_execute_->launchKernel(*function_hessian, grid, block_hessian, 0,
                                  parameters_hessian1);

    unsigned int current_nr_terms_x = max_nr_terms;
    for (unsigned int term_offset_x = last_term_y;
         term_offset_x < (unsigned int)nr_terms;
         term_offset_x += current_nr_terms_x) {
      unsigned int last_term_x =
          min(nr_terms, term_offset_x + current_nr_terms_x);
      current_nr_terms_x = last_term_x - term_offset_x;

      // Compute sums2 (horizontal offset)
      const void* parameters_sums[] = {&nr_polarizations,
                                       &subgrid_size,
                                       &image_size,
                                       &total_nr_timesteps,
                                       &nr_channels,
                                       &nr_stations,
                                       &term_offset_x,
                                       &current_nr_terms_x,
                                       &nr_terms,
                                       d_uvw.data(),
                                       d_wavenumbers.data(),
                                       d_aterm.data(),
                                       d_aterm_derivatives.data(),
                                       d_aterm_indices.data(),
                                       d_metadata.data(),
                                       d_subgrid.data(),
                                       d_sums2.data(),
                                       d_lmnp.data()};
      stream_execute_->launchKernel(*function_sums, grid, block, 0,
                                    parameters_sums);

      // Compute gradient (horizontal offset)
      if (term_offset_y == 0) {
        const void* parameters_gradient[] = {&nr_polarizations,
                                             &subgrid_size,
                                             &image_size,
                                             &total_nr_timesteps,
                                             &nr_channels,
                                             &nr_stations,
                                             &term_offset_x,
                                             &current_nr_terms_x,
                                             &nr_terms,
                                             d_uvw.data(),
                                             d_wavenumbers.data(),
                                             d_visibilities.data(),
                                             d_weights.data(),
                                             d_aterm.data(),
                                             d_aterm_derivatives.data(),
                                             d_aterm_indices.data(),
                                             d_metadata.data(),
                                             d_subgrid.data(),
                                             d_sums2.data(),
                                             d_lmnp.data(),
                                             d_gradient.data(),
                                             d_residual.data()};
        stream_execute_->launchKernel(*function_gradient, grid, block, 0,
                                      parameters_gradient);
      }

      // Compute hessian (horizontal offset)
      const void* parameters_hessian2[] = {
          &nr_polarizations, &total_nr_timesteps,    &nr_channels,
          &term_offset_y,    &term_offset_x,         &nr_terms,
          d_weights.data(),  d_aterm_indices.data(), d_metadata.data(),
          d_sums1.data(),    d_sums2.data(),         d_hessian.data()};
      dim3 block_hessian(current_nr_terms_x, current_nr_terms_y);
      stream_execute_->launchKernel(*function_hessian, grid, block_hessian, 0,
                                    parameters_hessian2);
    }
  }
  end_measurement(data);
}

void InstanceCUDA::launch_grid_fft(cu::DeviceMemory& d_data, int batch,
                                   long grid_size, DomainAtoDomainB direction) {
  cu::ScopedContext scc(*context_);

  int sign =
      (direction == FourierDomainToImageDomain) ? CUFFT_INVERSE : CUFFT_FORWARD;

  // Plan FFT
  cufft::C2C_2D plan(*context_, grid_size, grid_size);
  plan.setStream(*stream_execute_);

  // Enqueue start of measurement
  UpdateData* data =
      get_update_data(get_event(), *power_meter_, report_, Report::grid_fft);
  start_measurement(data);

#if ENABLE_REPEAT_KERNELS
  for (int i = 0; i < NR_REPETITIONS_GRID_FFT; i++) {
#endif

    // Enqueue fft for the entire batch
    for (int i = 0; i < batch; i++) {
      cufftComplex* data_ptr =
          reinterpret_cast<cufftComplex*>(static_cast<CUdeviceptr>(d_data));
      data_ptr += i * grid_size * grid_size;
      plan.execute(data_ptr, data_ptr, sign);
    }

#if ENABLE_REPEAT_KERNELS
  }
#endif

  // Enqueue end of measurement
  end_measurement(data);
}

void InstanceCUDA::plan_subgrid_fft(size_t size, size_t nr_polarizations) {
  // Force plan (re-)creation if subgrid size changed
  if (!fft_plan_subgrid_ || size != fft_subgrid_size_) {
    fft_subgrid_batch_ = kFftSubgridBatch;
    fft_plan_subgrid_.reset();
    fft_subgrid_size_ = size;
  } else {
    // The subgrid fft was initialized before
    return;
  }

  // Amount of device memory free (with a small safety margin)
  const size_t bytes_free = get_free_memory() * 0.95;

  // Amount of device memory required for temporary subgrids buffer and the FFT
  // plan
  const size_t bytes_required =
      2 * auxiliary::sizeof_subgrids(fft_subgrid_batch_, fft_subgrid_size_,
                                     nr_polarizations);

  // Compute the actual subgrid batch size to use
  const size_t fft_subgrid_batch_max = bytes_free / bytes_required;
  fft_subgrid_batch_ = std::min(fft_subgrid_batch_, fft_subgrid_batch_max);

  try {
    // Plan fft
    const size_t stride = 1;
    const size_t dist = size * size;
    fft_plan_subgrid_.reset(
        new cufft::C2C_2D(*context_, size, size, stride, dist,
                          fft_subgrid_batch_ * nr_polarizations));
    fft_plan_subgrid_->setStream(*stream_execute_);

    // Allocate temporary subgrid buffer
    const size_t sizeof_subgrids = auxiliary::sizeof_subgrids(
        fft_subgrid_batch_, fft_subgrid_size_, nr_polarizations);
    device_subgrid_fft_.reset(new cu::DeviceMemory(*context_, sizeof_subgrids));
  } catch (std::exception& e) {
    // Even though we tried to stay within the amount of available device
    // memory, allocating the fft plan or temporary subgrids buffer failed.
    std::stringstream message;
    message << __func__ << ": could not plan subgrid-fft for size = " << size
            << ", with " << bytes_free << " bytes of device memory available."
            << "(" << e.what() << ")" << std::endl;
    throw std::runtime_error(message.str());
  }
}

void InstanceCUDA::launch_subgrid_fft(cu::DeviceMemory& d_data,
                                      unsigned int nr_subgrids,
                                      unsigned int nr_polarizations,
                                      DomainAtoDomainB direction) {
  cufftComplex* data_ptr =
      reinterpret_cast<cufftComplex*>(static_cast<CUdeviceptr>(d_data));
  int sign =
      (direction == FourierDomainToImageDomain) ? CUFFT_INVERSE : CUFFT_FORWARD;

  cu::ScopedContext scc(*context_);

  // Enqueue start of measurement
  UpdateData* data =
      get_update_data(get_event(), *power_meter_, report_, Report::subgrid_fft);
  start_measurement(data);

  // Execute fft in batches
  for (unsigned s = 0; (s + fft_subgrid_batch_) <= nr_subgrids;
       s += fft_subgrid_batch_) {
    fft_plan_subgrid_->execute(data_ptr, data_ptr, sign);
    data_ptr += fft_subgrid_size_ * fft_subgrid_size_ * nr_polarizations *
                fft_subgrid_batch_;
  }

  // Check for remainder
  unsigned int fft_subgrid_remainder = nr_subgrids % fft_subgrid_batch_;
  if (fft_subgrid_remainder > 0) {
    auto sizeof_subgrids = auxiliary::sizeof_subgrids(
        fft_subgrid_remainder, fft_subgrid_size_, nr_polarizations);
    stream_execute_->memcpyDtoDAsync(*device_subgrid_fft_,
                                     reinterpret_cast<CUdeviceptr>(data_ptr),
                                     sizeof_subgrids);
    cufftComplex* tmp_ptr = reinterpret_cast<cufftComplex*>(
        static_cast<CUdeviceptr>(*device_subgrid_fft_));
    fft_plan_subgrid_->execute(tmp_ptr, tmp_ptr, sign);
    stream_execute_->memcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(data_ptr),
                                     reinterpret_cast<CUdeviceptr>(tmp_ptr),
                                     sizeof_subgrids);
  }

  // Enqueue end of measurement
  end_measurement(data);
}

void InstanceCUDA::launch_fft_shift(cu::DeviceMemory& d_data, int batch,
                                    long size, std::complex<float> scale) {
  const void* parameters[] = {&size, d_data.data(), &scale};

  dim3 grid(batch, ceil(size / 2.0));
  dim3 block(128);

  UpdateData* data =
      get_update_data(get_event(), *power_meter_, report_, Report::fft_shift);
  start_measurement(data);
  stream_execute_->launchKernel(*function_fft_shift_, grid, block, 0,
                                parameters);
  end_measurement(data);
}

void InstanceCUDA::launch_adder(int nr_subgrids, int nr_polarizations,
                                long grid_size, int subgrid_size,
                                cu::DeviceMemory& d_metadata,
                                cu::DeviceMemory& d_subgrid,
                                cu::DeviceMemory& d_grid) {
  const bool enable_tiling = false;
  const void* parameters[] = {
      &nr_polarizations, &grid_size,    &subgrid_size, d_metadata.data(),
      d_subgrid.data(),  d_grid.data(), &enable_tiling};
  dim3 grid(nr_subgrids);
  dim3 block(128);
  UpdateData* data =
      get_update_data(get_event(), *power_meter_, report_, Report::adder);
  start_measurement(data);
#if ENABLE_REPEAT_KERNELS
  for (int i = 0; i < NR_REPETITIONS_ADDER; i++)
#endif
    stream_execute_->launchKernel(*function_adder_, grid, block, 0, parameters);
  end_measurement(data);
}

void InstanceCUDA::launch_adder_unified(int nr_subgrids, long grid_size,
                                        int subgrid_size,
                                        cu::DeviceMemory& d_metadata,
                                        cu::DeviceMemory& d_subgrid,
                                        void* u_grid) {
  bool enable_tiling = true;
  const int nr_polarizations = 4;
  const void* parameters[] = {
      &nr_polarizations, &grid_size, &subgrid_size, d_metadata.data(),
      d_subgrid.data(),  &u_grid,    &enable_tiling};
  dim3 grid(nr_subgrids);
  dim3 block(128);
  UpdateData* data =
      get_update_data(get_event(), *power_meter_, report_, Report::adder);
  start_measurement(data);
  stream_execute_->launchKernel(*function_adder_, grid, block, 0, parameters);
  end_measurement(data);
}

void InstanceCUDA::launch_splitter(int nr_subgrids, int nr_polarizations,
                                   long grid_size, int subgrid_size,
                                   cu::DeviceMemory& d_metadata,
                                   cu::DeviceMemory& d_subgrid,
                                   cu::DeviceMemory& d_grid) {
  const bool enable_tiling = false;
  const void* parameters[] = {
      &nr_polarizations, &grid_size,    &subgrid_size, d_metadata.data(),
      d_subgrid.data(),  d_grid.data(), &enable_tiling};
  dim3 grid(nr_subgrids);
  dim3 block(128);
  UpdateData* data =
      get_update_data(get_event(), *power_meter_, report_, Report::splitter);
  start_measurement(data);
#if ENABLE_REPEAT_KERNELS
  for (int i = 0; i < NR_REPETITIONS_ADDER; i++)
#endif
    stream_execute_->launchKernel(*function_splitter_, grid, block, 0,
                                  parameters);
  end_measurement(data);
}

void InstanceCUDA::launch_splitter_unified(int nr_subgrids, long grid_size,
                                           int subgrid_size,
                                           cu::DeviceMemory& d_metadata,
                                           cu::DeviceMemory& d_subgrid,
                                           void* u_grid) {
  const bool enable_tiling = true;
  const int nr_polarizations = 4;
  const void* parameters[] = {
      &nr_polarizations, &grid_size, &subgrid_size, d_metadata.data(),
      d_subgrid.data(),  &u_grid,    &enable_tiling};
  dim3 grid(nr_subgrids);
  dim3 block(128);
  UpdateData* data =
      get_update_data(get_event(), *power_meter_, report_, Report::splitter);
  start_measurement(data);
  stream_execute_->launchKernel(*function_splitter_, grid, block, 0,
                                parameters);
  end_measurement(data);
}

void InstanceCUDA::launch_scaler(int nr_subgrids, int nr_polarizations,
                                 int subgrid_size,
                                 cu::DeviceMemory& d_subgrid) {
  const void* parameters[] = {&nr_polarizations, &subgrid_size,
                              d_subgrid.data()};
  dim3 grid(nr_subgrids);
  dim3 block(128);
  UpdateData* data =
      get_update_data(get_event(), *power_meter_, report_, Report::fft_scale);
  start_measurement(data);
  stream_execute_->launchKernel(*function_scaler_, grid, block, 0, parameters);
  end_measurement(data);
}

void InstanceCUDA::launch_copy_tiles(
    unsigned int nr_polarizations, unsigned int nr_tiles,
    unsigned int src_tile_size, unsigned int dst_tile_size,
    cu::DeviceMemory& d_src_tile_ids, cu::DeviceMemory& d_dst_tile_ids,
    cu::DeviceMemory& d_src_tiles, cu::DeviceMemory& d_dst_tiles) {
  const void* parameters[] = {&src_tile_size,        &dst_tile_size,
                              d_src_tile_ids.data(), d_dst_tile_ids.data(),
                              d_src_tiles.data(),    d_dst_tiles.data()};
  dim3 grid(nr_polarizations, nr_tiles);
  dim3 block(128);
  stream_execute_->launchKernel(*functions_wtiling_[0], grid, block, 0,
                                parameters);
}

void InstanceCUDA::launch_apply_phasor_to_wtiles(
    unsigned int nr_polarizations, unsigned int nr_tiles, float image_size,
    float w_step, unsigned int tile_size, cu::DeviceMemory& d_tiles,
    cu::DeviceMemory& d_shift, cu::DeviceMemory& d_tile_coordinates, int sign) {
  const void* parameters[] = {&image_size,    &w_step,
                              &tile_size,     d_tiles.data(),
                              d_shift.data(), d_tile_coordinates.data(),
                              &sign};
  dim3 grid(nr_polarizations, nr_tiles);
  dim3 block(128);
  stream_execute_->launchKernel(*functions_wtiling_[1], grid, block, 0,
                                parameters);
}

void InstanceCUDA::launch_adder_subgrids_to_wtiles(
    int nr_subgrids, int nr_polarizations, long grid_size, int subgrid_size,
    int tile_size, int subgrid_offset, cu::DeviceMemory& d_metadata,
    cu::DeviceMemory& d_subgrid, cu::DeviceMemory& d_tiles,
    std::complex<float> scale) {
  const void* parameters[] = {
      &nr_polarizations, &grid_size,      &subgrid_size,
      &tile_size,        &subgrid_offset, d_metadata.data(),
      d_subgrid.data(),  d_tiles.data(),  &scale};
  dim3 grid(nr_subgrids);
  dim3 block(128);
  stream_execute_->launchKernel(*functions_wtiling_[2], grid, block, 0,
                                parameters);
}

void InstanceCUDA::launch_adder_wtiles_to_grid(
    int nr_polarizations, int nr_tiles, long grid_size, int tile_size,
    int padded_tile_size, cu::DeviceMemory& d_tile_ids,
    cu::DeviceMemory& d_tile_coordinates, cu::DeviceMemory& d_tiles,
    void* u_grid) {
  const void* parameters[] = {&grid_size,
                              &tile_size,
                              &padded_tile_size,
                              d_tile_ids.data(),
                              d_tile_coordinates.data(),
                              d_tiles.data(),
                              &u_grid};
  dim3 grid(nr_polarizations, nr_tiles);
  dim3 block(128);
  stream_execute_->launchKernel(*functions_wtiling_[3], grid, block, 0,
                                parameters);
}

void InstanceCUDA::launch_splitter_subgrids_from_wtiles(
    int nr_subgrids, int nr_polarizations, long grid_size, int subgrid_size,
    int tile_size, int subgrid_offset, cu::DeviceMemory& d_metadata,
    cu::DeviceMemory& d_subgrid, cu::DeviceMemory& d_tiles) {
  const void* parameters[] = {
      &nr_polarizations, &grid_size,        &subgrid_size,    &tile_size,
      &subgrid_offset,   d_metadata.data(), d_subgrid.data(), d_tiles.data()};
  dim3 grid(nr_subgrids);
  dim3 block(128);
  stream_execute_->launchKernel(*functions_wtiling_[4], grid, block, 0,
                                parameters);
}

void InstanceCUDA::launch_splitter_wtiles_from_grid(
    int nr_polarizations, int nr_tiles, long grid_size, int tile_size,
    int padded_tile_size, cu::DeviceMemory& d_tile_ids,
    cu::DeviceMemory& d_tile_coordinates, cu::DeviceMemory& d_tiles,
    void* u_grid) {
  const void* parameters[] = {&grid_size,
                              &tile_size,
                              &padded_tile_size,
                              d_tile_ids.data(),
                              d_tile_coordinates.data(),
                              d_tiles.data(),
                              &u_grid};
  dim3 grid(nr_polarizations, nr_tiles);
  dim3 block(128);
  stream_execute_->launchKernel(*functions_wtiling_[5], grid, block, 0,
                                parameters);
}

void InstanceCUDA::launch_adder_wtiles_to_patch(
    int nr_polarizations, int nr_tiles, long grid_size, int tile_size,
    int padded_tile_size, int patch_size, idg::Coordinate patch_coordinate,
    cu::DeviceMemory& d_tile_ids, cu::DeviceMemory& d_tile_coordinates,
    cu::DeviceMemory& d_tiles, cu::DeviceMemory& d_patch) {
  const void* parameters[] = {&nr_tiles,         &grid_size,
                              &tile_size,        &padded_tile_size,
                              &patch_size,       &patch_coordinate,
                              d_tile_ids.data(), d_tile_coordinates.data(),
                              d_tiles.data(),    d_patch.data()};
  dim3 grid(nr_polarizations, patch_size);
  dim3 block(128);
  stream_execute_->launchKernel(*functions_wtiling_[6], grid, block, 0,
                                parameters);
}

void InstanceCUDA::launch_splitter_wtiles_from_patch(
    int nr_polarizations, int nr_tiles, long grid_size, int tile_size,
    int padded_tile_size, int patch_size, idg::Coordinate patch_coordinate,
    cu::DeviceMemory& d_tile_ids, cu::DeviceMemory& d_tile_coordinates,
    cu::DeviceMemory& d_tiles, cu::DeviceMemory& d_patch) {
  const void* parameters[] = {&nr_tiles,         &grid_size,
                              &tile_size,        &padded_tile_size,
                              &patch_size,       &patch_coordinate,
                              d_tile_ids.data(), d_tile_coordinates.data(),
                              d_tiles.data(),    d_patch.data()};
  dim3 grid(nr_polarizations, patch_size);
  dim3 block(128);
  stream_execute_->launchKernel(*functions_wtiling_[7], grid, block, 0,
                                parameters);
}

typedef struct {
  int nr_polarizations;
  int nr_timesteps;
  int nr_subgrids;
  std::shared_ptr<Report> report;
} ReportData;

void report_job(CUstream, CUresult, void* userData) {
  ReportData* data = static_cast<ReportData*>(userData);
  int nr_polarizations = data->nr_polarizations;
  int nr_correlations = nr_polarizations == 4 ? 4 : 2;
  int nr_timesteps = data->nr_timesteps;
  int nr_subgrids = data->nr_subgrids;
  data->report->print(nr_correlations, nr_timesteps, nr_subgrids);
  delete data;
}

ReportData* get_report_data(int nr_polarizations, int nr_timesteps,
                            int nr_subgrids, std::shared_ptr<Report> report) {
  ReportData* data = new ReportData();
  data->nr_polarizations = nr_polarizations;
  data->nr_timesteps = nr_timesteps;
  data->nr_subgrids = nr_subgrids;
  data->report = report;
  return data;
}

void InstanceCUDA::enqueue_report(cu::Stream& stream, int nr_polarizations,
                                  int nr_timesteps, int nr_subgrids) {
  ReportData* data =
      get_report_data(nr_polarizations, nr_timesteps, nr_subgrids, report_);
  stream.addCallback((CUstreamCallback)&report_job, data);
}

/*
 * Event destructor
 */
void InstanceCUDA::free_events() { events.clear(); }

/*
 * FFT plan destructor
 */
void InstanceCUDA::free_subgrid_fft() {
  fft_subgrid_batch_ = 0;
  fft_subgrid_size_ = 0;
  fft_plan_subgrid_.reset();
  device_subgrid_fft_.reset();
}

/*
 * Reset device
 */
void InstanceCUDA::reset() {
  stream_execute_.reset();
  stream_htod_.reset();
  stream_dtoh_.reset();
  context_.reset(new cu::Context(*device_));
  stream_execute_.reset(new cu::Stream(*context_));
  stream_htod_.reset(new cu::Stream(*context_));
  stream_dtoh_.reset(new cu::Stream(*context_));
}

/*
 * Device interface
 */
void InstanceCUDA::print_device_memory_info() const {
#if defined(DEBUG)
  std::cout << "InstanceCUDA::" << __func__ << std::endl;
#endif
  cu::ScopedContext scc(*context_);
  auto memory_total =
      device_->get_total_memory() / (float(1024) * 1024 * 1024);  // GBytes
  auto memory_free =
      device_->get_free_memory() / (float(1024) * 1024 * 1024);  // GBytes
  auto memory_used = memory_total - memory_free;
  std::clog << "Device memory -> ";
  std::clog << "total: " << memory_total << " Gb, ";
  std::clog << "used: " << memory_used << " Gb, ";
  std::clog << "free: " << memory_free << " Gb" << std::endl;
}

size_t InstanceCUDA::get_free_memory() const {
  cu::ScopedContext scc(*context_);
  return device_->get_free_memory();
}

size_t InstanceCUDA::get_total_memory() const {
  cu::ScopedContext scc(*context_);
  return device_->get_total_memory();
}

template <CUdevice_attribute attribute>
int InstanceCUDA::get_attribute() const {
  cu::ScopedContext scc(*context_);
  return device_->get_attribute<attribute>();
}

}  // end namespace idg::kernel::cuda
