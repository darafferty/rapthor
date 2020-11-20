// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <iomanip>  // setprecision

#include "InstanceCUDA.h"
#include "PowerRecord.h"

using namespace idg::kernel;
using namespace powersensor;

#define NR_CORRELATIONS 4

/*
 * Option to enable repeated kernel invocations
 * this is used to measure energy consumpton
 * using a low-resolution power measurement (NVML)
 */
#define ENABLE_REPEAT_KERNELS 0
#define NR_REPETITIONS_GRIDDER 10
#define NR_REPETITIONS_ADDER 50
#define NR_REPETITIONS_GRID_FFT 500

/*
 * Use custom FFT kernel
 */
#define USE_CUSTOM_FFT 0

namespace idg {
namespace kernel {
namespace cuda {

// Constructor
InstanceCUDA::InstanceCUDA(ProxyInfo& info, int device_nr, int device_id)
    : KernelsInstance(), mInfo(info) {
#if defined(DEBUG)
  std::cout << __func__ << std::endl;
#endif

  // Initialize members
  device.reset(new cu::Device(device_id));
  context.reset(new cu::Context(*device));
  context->setCurrent();
  executestream.reset(new cu::Stream());
  htodstream.reset(new cu::Stream());
  dtohstream.reset(new cu::Stream());

  // Set kernel parameters
  set_parameters();

  // Compile kernels
  compile_kernels();

  // Load kernels
  load_kernels();

  // Initialize power sensor
  powerSensor = get_power_sensor(sensor_device, device_nr);
}

// Destructor
InstanceCUDA::~InstanceCUDA() {
  context->setCurrent();
  free_host_memory();
  free_device_memory();
  free_fft_plans();
  mModules.clear();
  executestream.reset();
  htodstream.reset();
  dtohstream.reset();
  context->reset();
  device.reset();
  context.reset();
  delete powerSensor;
}

/*
    Compilation
*/
std::string InstanceCUDA::get_compiler_flags() {
  // Constants
  std::stringstream flags_constants;
  flags_constants << "-DNR_POLARIZATIONS=" << NR_CORRELATIONS;

  // CUDA specific flags
  std::stringstream flags_cuda;
  flags_cuda << "-use_fast_math ";
#if defined(CUDA_KERNEL_DEBUG)
  flags_cuda << " -G ";
#else
  flags_cuda << "-lineinfo ";
#endif
  flags_cuda << "-src-in-ptx";

  // Device specific flags
  int capability = (*device).get_capability();
  std::stringstream flags_device;
  flags_device << "-arch=sm_" << capability;

  // Include flags
  std::stringstream flags_includes;
  flags_includes << "-I" << IDG_INSTALL_DIR << "/include";

  // Combine flags
  std::string flags = " " + flags_cuda.str() + " " + flags_device.str() + " " +
                      flags_constants.str() + " " + flags_includes.str();
  return flags;
}

cu::Module* InstanceCUDA::compile_kernel(std::string& flags, std::string& src,
                                         std::string& bin) {
  context->setCurrent();

  // Create a string with the full path to the cubin file "kernel.cubin"
  std::string lib = mInfo.get_path_to_lib() + "/" + bin;

  // Create a string for all sources that are combined
  std::string source = mInfo.get_path_to_src() + "/" + src;

  // Call the compiler
  cu::Source(source.c_str()).compile(lib.c_str(), flags.c_str());

  // Create module
  return new cu::Module(lib.c_str());
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
  std::stringstream flags_gridder;
  flags_gridder << flags_common;
  flags_gridder << " -DBATCH_SIZE=" << batch_gridder;
  flags.push_back(flags_gridder.str());

  // Degridder
  src.push_back("KernelDegridder.cu");
  cubin.push_back("Degridder.cubin");
  std::stringstream flags_degridder;
  flags_degridder << flags_common;
  flags_degridder << " -DBATCH_SIZE=" << batch_degridder;
  flags.push_back(flags_degridder.str());

  // Scaler
  src.push_back("KernelScaler.cu");
  cubin.push_back("Scaler.cubin");
  flags.push_back(flags_common);

  // Adder
  src.push_back("KernelAdder.cu");
  cubin.push_back("Adder.cubin");
  std::stringstream flags_adder;
  flags_adder << flags_common;
  flags_adder << " -DTILE_SIZE_GRID=" << tile_size_grid;
  flags.push_back(flags_adder.str());

  // Splitter
  src.push_back("KernelSplitter.cu");
  cubin.push_back("Splitter.cubin");
  std::stringstream flags_splitter;
  flags_splitter << flags_common;
  flags_splitter << " -DTILE_SIZE_GRID=" << tile_size_grid;
  flags.push_back(flags_splitter.str());

  // Calibrate
  src.push_back("KernelCalibrate.cu");
  cubin.push_back("Calibrate.cubin");
  flags.push_back(flags_common);

// FFT
#if USE_CUSTOM_FFT
  src.push_back("KernelFFT.cu");
  cubin.push_back("FFT.cubin");
  flags.push_back(flags_common);
#endif

  // Compile all kernels
  for (unsigned i = 0; i < src.size(); i++) {
    mModules.push_back(std::unique_ptr<cu::Module>());
  }
#pragma omp parallel for
  for (unsigned i = 0; i < src.size(); i++) {
    mModules[i].reset(compile_kernel(flags[i], src[i], cubin[i]));
  }
}

void InstanceCUDA::load_kernels() {
  CUfunction function;
  unsigned found = 0;

  // Load gridder function
  if (cuModuleGetFunction(&function, *mModules[0], name_gridder.c_str()) ==
      CUDA_SUCCESS) {
    function_gridder.reset(new cu::Function(function));
    found++;
  }

  // Load degridder function
  if (cuModuleGetFunction(&function, *mModules[1], name_degridder.c_str()) ==
      CUDA_SUCCESS) {
    function_degridder.reset(new cu::Function(function));
    found++;
  }

  // Load scalar function
  if (cuModuleGetFunction(&function, *mModules[2], name_scaler.c_str()) ==
      CUDA_SUCCESS) {
    function_scaler.reset(new cu::Function(function));
    found++;
  }

  // Load adder function
  if (cuModuleGetFunction(&function, *mModules[3], name_adder.c_str()) ==
      CUDA_SUCCESS) {
    function_adder.reset(new cu::Function(function));
    found++;
  }

  // Load splitter function
  if (cuModuleGetFunction(&function, *mModules[4], name_splitter.c_str()) ==
      CUDA_SUCCESS) {
    function_splitter.reset(new cu::Function(function));
    found++;
  }

  // Load calibration functions
  if (cuModuleGetFunction(&function, *mModules[5],
                          name_calibrate_lmnp.c_str()) == CUDA_SUCCESS) {
    functions_calibrate.push_back(
        std::unique_ptr<cu::Function>(new cu::Function(function)));
    found++;
  }
  if (cuModuleGetFunction(&function, *mModules[5],
                          name_calibrate_sums.c_str()) == CUDA_SUCCESS) {
    functions_calibrate.push_back(
        std::unique_ptr<cu::Function>(new cu::Function(function)));
  }
  if (cuModuleGetFunction(&function, *mModules[5],
                          name_calibrate_gradient.c_str()) == CUDA_SUCCESS) {
    functions_calibrate.push_back(
        std::unique_ptr<cu::Function>(new cu::Function(function)));
  }
  if (cuModuleGetFunction(&function, *mModules[5],
                          name_calibrate_hessian.c_str()) == CUDA_SUCCESS) {
    functions_calibrate.push_back(
        std::unique_ptr<cu::Function>(new cu::Function(function)));
  }

// Load FFT function
#if USE_CUSTOM_FFT
  if (cuModuleGetFunction(&function, *mModules[6], name_fft.c_str()) ==
      CUDA_SUCCESS) {
    function_fft.reset(new cu::Function(function));
    found++;
  }
#endif

  // Verify that all functions are found
  if (found != mModules.size()) {
    std::cerr << "Incorrect number of functions found: " << found
              << " != " << mModules.size() << std::endl;
    exit(EXIT_FAILURE);
  }
}

void InstanceCUDA::set_parameters_kepler() {
  batch_gridder = 192;
  batch_degridder = 192;
}

void InstanceCUDA::set_parameters_maxwell() {
  batch_gridder = 384;
  batch_degridder = 512;
}

void InstanceCUDA::set_parameters_gp100() {
  batch_gridder = 256;
  batch_degridder = 128;
}

void InstanceCUDA::set_parameters_pascal() {
  batch_gridder = 384;
  batch_degridder = 512;
}

void InstanceCUDA::set_parameters_volta() {
  batch_gridder = 128;
  batch_degridder = 256;
}

void InstanceCUDA::set_parameters_default() {
  block_gridder = dim3(128);
  block_degridder = dim3(128);
  block_calibrate = dim3(128);
  block_adder = dim3(128);
  block_splitter = dim3(128);
  block_scaler = dim3(128);
  tile_size_grid = 128;
}

void InstanceCUDA::set_parameters() {
#if defined(DEBUG)
  std::cout << __func__ << std::endl;
#endif

  int capability = (*device).get_capability();

  set_parameters_default();

  if (capability >= 70) {
    set_parameters_volta();
  } else if (capability >= 61) {
    set_parameters_pascal();
  } else if (capability == 60) {
    set_parameters_gp100();
  } else if (capability >= 50) {
    set_parameters_maxwell();
  } else if (capability >= 30) {
    set_parameters_kepler();
  } else {
    // IDG has never been tested with pre-Kepler GPUs
    set_parameters_kepler();
  }

  // Override parameters from environment
  char* cstr_batch_size = getenv("BATCHSIZE");
  if (cstr_batch_size) {
    auto batch_size = atoi(cstr_batch_size);
    batch_gridder = batch_size;
    batch_degridder = batch_size;
  }
  char* cstr_block_size = getenv("BLOCKSIZE");
  if (cstr_block_size) {
    auto block_size = atoi(cstr_block_size);
    block_gridder = dim3(block_size);
    block_degridder = dim3(block_size);
  }
}

std::ostream& operator<<(std::ostream& os, InstanceCUDA& d) {
  os << d.get_device().get_name() << std::endl;
  os << std::setprecision(2);
  os << std::fixed;
  d.get_context().setCurrent();

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

State InstanceCUDA::measure() { return powerSensor->read(); }

void InstanceCUDA::measure(PowerRecord& record, cu::Stream& stream) {
  record.sensor = powerSensor;
  record.enqueue(stream);
}

typedef struct {
  PowerRecord* start;
  PowerRecord* end;
  Report* report;
  void (Report::*update_report)(State&, State&);
} UpdateData;

UpdateData* get_update_data(PowerSensor* sensor, Report* report,
                            void (Report::*update_report)(State&, State&)) {
  UpdateData* data = new UpdateData();
  data->start = new PowerRecord(sensor);
  data->end = new PowerRecord(sensor);
  data->report = report;
  data->update_report = update_report;
  return data;
}

void update_report_callback(CUstream, CUresult, void* userData) {
  UpdateData* data = static_cast<UpdateData*>(userData);
  PowerRecord* start = data->start;
  PowerRecord* end = data->end;
  Report* report = data->report;
  (report->*data->update_report)(start->state, end->state);
  delete start;
  delete end;
  delete data;
}

void InstanceCUDA::start_measurement(void* ptr) {
  UpdateData* data = (UpdateData*)ptr;

  // Schedule the first measurement (prior to kernel execution)
  data->start->enqueue(*executestream);
}

void InstanceCUDA::end_measurement(void* ptr) {
  UpdateData* data = (UpdateData*)ptr;

  // Schedule the second measurement (after the kernel execution)
  data->end->enqueue(*executestream);

  // Afterwards, update the report according to the two measurements
  executestream->addCallback((CUstreamCallback)&update_report_callback, data);
}

void InstanceCUDA::launch_gridder(
    int time_offset, int nr_subgrids, int grid_size, int subgrid_size,
    float image_size, float w_step, int nr_channels, int nr_stations,
    cu::DeviceMemory& d_uvw, cu::DeviceMemory& d_wavenumbers,
    cu::DeviceMemory& d_visibilities, cu::DeviceMemory& d_spheroidal,
    cu::DeviceMemory& d_aterm, cu::DeviceMemory& d_aterm_indices,
    cu::DeviceMemory& d_avg_aterm_correction, cu::DeviceMemory& d_metadata,
    cu::DeviceMemory& d_subgrid) {
  const void* parameters[] = {&time_offset,    &grid_size,
                              &subgrid_size,   &image_size,
                              &w_step,         &nr_channels,
                              &nr_stations,    d_uvw,
                              d_wavenumbers,   d_visibilities,
                              d_spheroidal,    d_aterm,
                              d_aterm_indices, d_avg_aterm_correction,
                              d_metadata,      d_subgrid};

  dim3 grid(nr_subgrids);
  dim3 block(block_gridder);
  UpdateData* data =
      get_update_data(powerSensor, report, &Report::update_gridder);
  start_measurement(data);
#if ENABLE_REPEAT_KERNELS
  for (int i = 0; i < NR_REPETITIONS_GRIDDER; i++)
#endif
    executestream->launchKernel(*function_gridder, grid, block, 0, parameters);
  end_measurement(data);
}

void InstanceCUDA::launch_degridder(
    int time_offset, int nr_subgrids, int grid_size, int subgrid_size,
    float image_size, float w_step, int nr_channels, int nr_stations,
    cu::DeviceMemory& d_uvw, cu::DeviceMemory& d_wavenumbers,
    cu::DeviceMemory& d_visibilities, cu::DeviceMemory& d_spheroidal,
    cu::DeviceMemory& d_aterm, cu::DeviceMemory& d_aterm_indices,
    cu::DeviceMemory& d_metadata, cu::DeviceMemory& d_subgrid) {
  const void* parameters[] = {&time_offset,    &grid_size,   &subgrid_size,
                              &image_size,     &w_step,      &nr_channels,
                              &nr_stations,    d_uvw,        d_wavenumbers,
                              d_visibilities,  d_spheroidal, d_aterm,
                              d_aterm_indices, d_metadata,   d_subgrid};

  dim3 grid(nr_subgrids);
  dim3 block(block_degridder);
  UpdateData* data =
      get_update_data(powerSensor, report, &Report::update_degridder);
  start_measurement(data);
#if ENABLE_REPEAT_KERNELS
  for (int i = 0; i < NR_REPETITIONS_GRIDDER; i++)
#endif
    executestream->launchKernel(*function_degridder, grid, block, 0,
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
  dim3 block(block_calibrate);
  UpdateData* data =
      get_update_data(powerSensor, report, &Report::update_calibrate);
  start_measurement(data);

  // Get functions
  std::unique_ptr<cu::Function>& function_lmnp = functions_calibrate[0];
  std::unique_ptr<cu::Function>& function_sums = functions_calibrate[1];
  std::unique_ptr<cu::Function>& function_gradient = functions_calibrate[2];
  std::unique_ptr<cu::Function>& function_hessian = functions_calibrate[3];

  // Precompute l,m,n and phase offset
  const void* parameters_lmnp[] = {&grid_size, &subgrid_size, &image_size,
                                   &w_step,    &d_metadata,   &d_lmnp};
  executestream->launchKernel(*function_lmnp, grid, block, 0, parameters_lmnp);

  unsigned int max_nr_terms = 8;
  unsigned int current_nr_terms_y = max_nr_terms;
  for (unsigned int term_offset_y = 0; term_offset_y < (unsigned int)nr_terms;
       term_offset_y += current_nr_terms_y) {
    unsigned int last_term_y =
        min(nr_terms, term_offset_y + current_nr_terms_y);
    unsigned int current_nr_terms_y = last_term_y - term_offset_y;

    // Compute sums1
    const void* parameters_sums[] = {&subgrid_size,
                                     &image_size,
                                     &total_nr_timesteps,
                                     &nr_channels,
                                     &nr_stations,
                                     &term_offset_y,
                                     &current_nr_terms_y,
                                     &nr_terms,
                                     d_uvw,
                                     d_wavenumbers,
                                     d_aterm,
                                     d_aterm_derivatives,
                                     d_aterm_indices,
                                     d_metadata,
                                     d_subgrid,
                                     d_sums1,
                                     d_lmnp};
    executestream->launchKernel(*function_sums, grid, block, 0,
                                parameters_sums);

    // Compute gradient (diagonal)
    if (term_offset_y == 0) {
      const void* parameters_gradient[] = {&subgrid_size,
                                           &image_size,
                                           &total_nr_timesteps,
                                           &nr_channels,
                                           &nr_stations,
                                           &term_offset_y,
                                           &current_nr_terms_y,
                                           &nr_terms,
                                           d_uvw,
                                           d_wavenumbers,
                                           d_visibilities,
                                           d_weights,
                                           d_aterm,
                                           d_aterm_derivatives,
                                           d_aterm_indices,
                                           d_metadata,
                                           d_subgrid,
                                           d_sums1,
                                           d_lmnp,
                                           d_gradient,
                                           d_residual};
      executestream->launchKernel(*function_gradient, grid, block, 0,
                                  parameters_gradient);
    }

    // Compute hessian (diagonal)
    const void* parameters_hessian1[] = {&total_nr_timesteps,
                                         &nr_channels,
                                         &term_offset_y,
                                         &term_offset_y,
                                         &nr_terms,
                                         d_weights,
                                         d_aterm_indices,
                                         d_metadata,
                                         d_sums1,
                                         d_sums1,
                                         d_hessian};
    dim3 block_hessian(current_nr_terms_y, current_nr_terms_y);
    executestream->launchKernel(*function_hessian, grid, block_hessian, 0,
                                parameters_hessian1);

    unsigned int current_nr_terms_x = max_nr_terms;
    for (unsigned int term_offset_x = last_term_y;
         term_offset_x < (unsigned int)nr_terms;
         term_offset_x += current_nr_terms_x) {
      unsigned int last_term_x =
          min(nr_terms, term_offset_x + current_nr_terms_x);
      current_nr_terms_x = last_term_x - term_offset_x;

      // Compute sums2 (horizontal offset)
      const void* parameters_sums[] = {&subgrid_size,
                                       &image_size,
                                       &total_nr_timesteps,
                                       &nr_channels,
                                       &nr_stations,
                                       &term_offset_x,
                                       &current_nr_terms_x,
                                       &nr_terms,
                                       d_uvw,
                                       d_wavenumbers,
                                       d_aterm,
                                       d_aterm_derivatives,
                                       d_aterm_indices,
                                       d_metadata,
                                       d_subgrid,
                                       d_sums2,
                                       d_lmnp};
      executestream->launchKernel(*function_sums, grid, block, 0,
                                  parameters_sums);

      // Compute gradient (horizontal offset)
      if (term_offset_y == 0) {
        const void* parameters_gradient[] = {&subgrid_size,
                                             &image_size,
                                             &total_nr_timesteps,
                                             &nr_channels,
                                             &nr_stations,
                                             &term_offset_x,
                                             &current_nr_terms_x,
                                             &nr_terms,
                                             d_uvw,
                                             d_wavenumbers,
                                             d_visibilities,
                                             d_weights,
                                             d_aterm,
                                             d_aterm_derivatives,
                                             d_aterm_indices,
                                             d_metadata,
                                             d_subgrid,
                                             d_sums2,
                                             d_lmnp,
                                             d_gradient};
        executestream->launchKernel(*function_gradient, grid, block, 0,
                                    parameters_gradient);
      }

      // Compute hessian (horizontal offset)
      const void* parameters_hessian2[] = {&total_nr_timesteps,
                                           &nr_channels,
                                           &term_offset_y,
                                           &term_offset_x,
                                           &nr_terms,
                                           d_weights,
                                           d_aterm_indices,
                                           d_metadata,
                                           d_sums1,
                                           d_sums2,
                                           d_hessian};
      dim3 block_hessian(current_nr_terms_x, current_nr_terms_y);
      executestream->launchKernel(*function_hessian, grid, block_hessian, 0,
                                  parameters_hessian2);
    }
  }
  end_measurement(data);
}

void InstanceCUDA::launch_grid_fft(cu::DeviceMemory& d_data, int grid_size,
                                   DomainAtoDomainB direction) {
  int sign =
      (direction == FourierDomainToImageDomain) ? CUFFT_INVERSE : CUFFT_FORWARD;

  // Plan FFT
  if (grid_size != m_fft_grid_size) {
    if (m_fft_plan_grid) {
      m_fft_plan_grid.reset();
    }
    m_fft_grid_size = grid_size;
    m_fft_plan_grid.reset(new cufft::C2C_2D(grid_size, grid_size));
    m_fft_plan_grid->setStream(*executestream);
  }

  // Enqueue start of measurement
  UpdateData* data =
      get_update_data(powerSensor, report, &Report::update_grid_fft);
  start_measurement(data);

#if ENABLE_REPEAT_KERNELS
  for (int i = 0; i < NR_REPETITIONS_GRID_FFT; i++) {
#endif

    // Enqueue fft for every correlation
    for (unsigned i = 0; i < NR_CORRELATIONS; i++) {
      cufftComplex* data_ptr =
          reinterpret_cast<cufftComplex*>(static_cast<CUdeviceptr>(d_data));
      data_ptr += i * grid_size * grid_size;
      m_fft_plan_grid->execute(data_ptr, data_ptr, sign);
    }

#if ENABLE_REPEAT_KERNELS
  }
#endif

  // Enqueue end of measurement
  end_measurement(data);
}

void InstanceCUDA::plan_subgrid_fft(unsigned size, unsigned batch) {
#if USE_CUSTOM_FFT
  if (size == 32) {
    m_fft_subgrid_size = size;
    m_fft_subgrid_bulk = batch;
    return;
  }
#endif

  // Force plan (re-)creation if subgrid size changed
  if (size != m_fft_subgrid_size) {
    m_fft_subgrid_bulk = m_fft_subgrid_bulk_default;
    m_fft_plan_subgrid.reset();
    m_fft_subgrid_size = size;
  }

  while (m_fft_plan_subgrid == nullptr) {
    try {
      // Plan bulk fft
      unsigned stride = 1;
      unsigned dist = size * size;
      auto fft_plan = new cufft::C2C_2D(size, size, stride, dist,
                                        m_fft_subgrid_bulk * NR_CORRELATIONS);
      fft_plan->setStream(*executestream);
      m_fft_plan_subgrid.reset(fft_plan);
      auto sizeof_subgrids =
          auxiliary::sizeof_subgrids(m_fft_subgrid_bulk, m_fft_subgrid_size);
      d_fft_subgrid.reset(new cu::DeviceMemory(sizeof_subgrids));
    } catch (cufft::Error& e) {
      // bulk might be too large, try again using half the bulk size
      m_fft_subgrid_bulk /= 2;
      if (m_fft_subgrid_bulk > 0) {
        std::clog << __func__ << ": reducing subgrid-fft bulk size to: "
                  << m_fft_subgrid_bulk << std::endl;
      } else {
        std::cerr << __func__ << ": could not plan subgrid-fft." << std::endl;
        throw e;
      }
    }
  }
}

void InstanceCUDA::launch_subgrid_fft(cu::DeviceMemory& d_data,
                                      unsigned int nr_subgrids,
                                      DomainAtoDomainB direction) {
  cufftComplex* data_ptr =
      reinterpret_cast<cufftComplex*>(static_cast<CUdeviceptr>(d_data));
  int sign =
      (direction == FourierDomainToImageDomain) ? CUFFT_INVERSE : CUFFT_FORWARD;

#if USE_CUSTOM_FFT
  if (fft_subgrid_size == 32) {
    const void* parameters[] = {&data_ptr, &data_ptr, &sign};
    dim3 block(128);
    dim3 grid(NR_CORRELATIONS * fft_subgrid_batch);
    executestream->launchKernel(*function_fft, grid, block, 0, parameters);
    return;
  }
#endif

  // Enqueue start of measurement
  UpdateData* data =
      get_update_data(powerSensor, report, &Report::update_subgrid_fft);
  start_measurement(data);

  // Execute bulk subgrid fft
  unsigned s = 0;
  for (; (s + m_fft_subgrid_bulk) <= nr_subgrids; s += m_fft_subgrid_bulk) {
    m_fft_plan_subgrid->execute(data_ptr, data_ptr, sign);
    data_ptr += m_fft_subgrid_size * m_fft_subgrid_size * NR_CORRELATIONS *
                m_fft_subgrid_bulk;
  }

  // Check for remainder
  unsigned int fft_subgrid_remainder = nr_subgrids % m_fft_subgrid_bulk;
  if (fft_subgrid_remainder > 0) {
    auto sizeof_subgrids =
        auxiliary::sizeof_subgrids(fft_subgrid_remainder, m_fft_subgrid_size);
    executestream->memcpyDtoDAsync(*d_fft_subgrid, (CUdeviceptr)data_ptr,
                                   sizeof_subgrids);
    cufftComplex* tmp_ptr = reinterpret_cast<cufftComplex*>(
        static_cast<CUdeviceptr>(*d_fft_subgrid));
    m_fft_plan_subgrid->execute(tmp_ptr, tmp_ptr, sign);
    executestream->memcpyDtoDAsync((CUdeviceptr)data_ptr, (CUdeviceptr)tmp_ptr,
                                   sizeof_subgrids);
  }
  executestream->synchronize();

  // Enqueue end of measurement
  end_measurement(data);
}

void InstanceCUDA::launch_grid_fft_unified(unsigned long size,
                                           unsigned int batch,
                                           Array3D<std::complex<float>>& grid,
                                           DomainAtoDomainB direction) {
  int sign =
      (direction == FourierDomainToImageDomain) ? CUFFT_INVERSE : CUFFT_FORWARD;
  cufft::C2C_1D fft_plan_row(size, 1, 1, 1);
  cufft::C2C_1D fft_plan_col(size, size, 1, 1);

  for (unsigned i = 0; i < batch; i++) {
    // Execute 1D FFT over all columns
    for (unsigned col = 0; col < size; col++) {
      cufftComplex* ptr = (cufftComplex*)grid.data(i, col, 0);
      fft_plan_row.execute(ptr, ptr, sign);
    }

    // Execute 1D FFT over all rows
    for (unsigned row = 0; row < size; row++) {
      cufftComplex* ptr = (cufftComplex*)grid.data(i, 0, row);
      fft_plan_col.execute(ptr, ptr, sign);
    }
  }
}

void InstanceCUDA::launch_adder(int nr_subgrids, long grid_size,
                                int subgrid_size, cu::DeviceMemory& d_metadata,
                                cu::DeviceMemory& d_subgrid,
                                cu::DeviceMemory& d_grid) {
  const bool enable_tiling = false;
  const void* parameters[] = {&grid_size, &subgrid_size, d_metadata,
                              d_subgrid,  d_grid,        &enable_tiling};
  dim3 grid(nr_subgrids);
  UpdateData* data =
      get_update_data(powerSensor, report, &Report::update_adder);
  start_measurement(data);
  data->start->enqueue(*executestream);
#if ENABLE_REPEAT_KERNELS
  for (int i = 0; i < NR_REPETITIONS_ADDER; i++)
#endif
    executestream->launchKernel(*function_adder, grid, block_adder, 0,
                                parameters);
  end_measurement(data);
}

void InstanceCUDA::launch_adder_unified(int nr_subgrids, long grid_size,
                                        int subgrid_size,
                                        cu::DeviceMemory& d_metadata,
                                        cu::DeviceMemory& d_subgrid,
                                        void* u_grid) {
  const bool enable_tiling = true;
  const void* parameters[] = {&grid_size, &subgrid_size, d_metadata,
                              d_subgrid,  &u_grid,       &enable_tiling};
  dim3 grid(nr_subgrids);
  UpdateData* data =
      get_update_data(powerSensor, report, &Report::update_adder);
  start_measurement(data);
  executestream->launchKernel(*function_adder, grid, block_adder, 0,
                              parameters);
  end_measurement(data);
}

void InstanceCUDA::launch_splitter(int nr_subgrids, long grid_size,
                                   int subgrid_size,
                                   cu::DeviceMemory& d_metadata,
                                   cu::DeviceMemory& d_subgrid,
                                   cu::DeviceMemory& d_grid) {
  const bool enable_tiling = false;
  const void* parameters[] = {&grid_size, &subgrid_size, d_metadata,
                              d_subgrid,  d_grid,        &enable_tiling};
  dim3 grid(nr_subgrids);
  UpdateData* data =
      get_update_data(powerSensor, report, &Report::update_splitter);
  start_measurement(data);
#if ENABLE_REPEAT_KERNELS
  for (int i = 0; i < NR_REPETITIONS_ADDER; i++)
#endif
    executestream->launchKernel(*function_splitter, grid, block_splitter, 0,
                                parameters);
  end_measurement(data);
}

void InstanceCUDA::launch_splitter_unified(int nr_subgrids, long grid_size,
                                           int subgrid_size,
                                           cu::DeviceMemory& d_metadata,
                                           cu::DeviceMemory& d_subgrid,
                                           void* u_grid) {
  const bool enable_tiling = true;
  const void* parameters[] = {&grid_size, &subgrid_size, d_metadata,
                              d_subgrid,  &u_grid,       &enable_tiling};
  dim3 grid(nr_subgrids);
  UpdateData* data =
      get_update_data(powerSensor, report, &Report::update_splitter);
  start_measurement(data);
  executestream->launchKernel(*function_splitter, grid, block_splitter, 0,
                              parameters);
  end_measurement(data);
}

void InstanceCUDA::launch_scaler(int nr_subgrids, int subgrid_size,
                                 cu::DeviceMemory& d_subgrid) {
  const void* parameters[] = {&subgrid_size, d_subgrid};
  dim3 grid(nr_subgrids);
  UpdateData* data =
      get_update_data(powerSensor, report, &Report::update_scaler);
  start_measurement(data);
  executestream->launchKernel(*function_scaler, grid, block_scaler, 0,
                              parameters);
  end_measurement(data);
}

typedef struct {
  int nr_timesteps;
  int nr_subgrids;
  Report* report;
} ReportData;

void report_job(CUstream, CUresult, void* userData) {
  ReportData* data = static_cast<ReportData*>(userData);
  int nr_timesteps = data->nr_timesteps;
  int nr_subgrids = data->nr_subgrids;
  Report* report = data->report;
  report->print(nr_timesteps, nr_subgrids);
  delete data;
}

ReportData* get_report_data(int nr_timesteps, int nr_subgrids, Report* report) {
  ReportData* data = new ReportData();
  data->nr_timesteps = nr_timesteps;
  data->nr_subgrids = nr_subgrids;
  data->report = report;
  return data;
}

void InstanceCUDA::enqueue_report(cu::Stream& stream, int nr_timesteps,
                                  int nr_subgrids) {
  ReportData* data = get_report_data(nr_timesteps, nr_subgrids, report);
  stream.addCallback((CUstreamCallback)&report_job, data);
}

void InstanceCUDA::copy_htoh(void* dst, void* src, size_t bytes) {
  char message[80];
  snprintf(message, 80, "copy_htoh(%lu)", bytes);
  cu::Marker marker(message, cu::Marker::red);
  marker.start();
  size_t batch = 1024 * 1024 * 1024;  // 1024 Mb
#pragma omp parallel for
  for (size_t i = 0; i < bytes; i += batch) {
    size_t n = i + batch < bytes ? batch : bytes - i;
    size_t src_ptr = (size_t)src + i;
    size_t dst_ptr = (size_t)dst + i;
    memcpy((void*)dst_ptr, (void*)src_ptr, n);
  }
  marker.end();
}

void InstanceCUDA::copy_dtoh(cu::Stream& stream, void* dst,
                             cu::DeviceMemory& src, size_t bytes) {
  char message[80];
  snprintf(message, 80, "copy_dtoh(%lu)", bytes);
  cu::Marker marker(message, cu::Marker::red);
  marker.start();
  size_t batch = 1024 * 1024 * 1024;  // 1024 Mb
  cu::HostMemory tmp(batch);
  for (size_t i = 0; i < bytes; i += batch) {
    size_t n = i + batch < bytes ? batch : bytes - i;
    size_t src_ptr = (size_t)src + i;
    size_t dst_ptr = (size_t)dst + i;
    stream.memcpyDtoHAsync((void*)tmp, (CUdeviceptr)src_ptr, n);
    stream.synchronize();
    memcpy((void*)dst_ptr, tmp, n);
  }
  marker.end();
}

void InstanceCUDA::copy_htod(cu::Stream& stream, cu::DeviceMemory& dst,
                             void* src, size_t bytes) {
  char message[80];
  snprintf(message, 80, "copy_htod(%lu)", bytes);
  cu::Marker marker(message, cu::Marker::red);
  marker.start();
  size_t batch = 1024 * 1024 * 1024;  // 1024 Mb
  cu::HostMemory tmp(batch);
  for (size_t i = 0; i < bytes; i += batch) {
    size_t n = i + batch < bytes ? batch : bytes - i;
    size_t src_ptr = (size_t)src + i;
    size_t dst_ptr = (size_t)dst + i;
    memcpy(tmp, (void*)src_ptr, n);
    stream.memcpyHtoDAsync((CUdeviceptr)dst_ptr, tmp, n);
    stream.synchronize();
  }
  marker.end();
}

/*
 *  Memory management per device
 *      Maintain one memory object per data structure
 */
template <typename T>
T* InstanceCUDA::reuse_memory(uint64_t size, std::unique_ptr<T>& memory) {
  if (!memory) {
    memory.reset(new T(size));
  } else {
    memory->resize(size);
  }
  return memory.get();
}

cu::DeviceMemory& InstanceCUDA::allocate_device_grid(size_t bytes) {
  return *reuse_memory(bytes, d_grid);
}

cu::HostMemory& InstanceCUDA::allocate_host_visibilities(size_t bytes) {
  return *reuse_memory(bytes, h_visibilities);
}

cu::HostMemory& InstanceCUDA::allocate_host_subgrids(size_t bytes) {
  return *reuse_memory(bytes, h_subgrids);
}

cu::HostMemory& InstanceCUDA::allocate_host_uvw(size_t bytes) {
  return *reuse_memory(bytes, h_uvw);
}

cu::DeviceMemory& InstanceCUDA::allocate_device_aterms(size_t bytes) {
  return *reuse_memory(bytes, d_aterms);
}

cu::DeviceMemory& InstanceCUDA::allocate_device_aterms_indices(size_t bytes) {
  return *reuse_memory(bytes, d_aterms_indices);
}

cu::DeviceMemory& InstanceCUDA::allocate_device_wavenumbers(size_t bytes) {
  return *reuse_memory(bytes, d_wavenumbers);
}

cu::DeviceMemory& InstanceCUDA::allocate_device_spheroidal(size_t bytes) {
  return *reuse_memory(bytes, d_spheroidal);
}

cu::DeviceMemory& InstanceCUDA::allocate_device_avg_aterm_correction(
    size_t bytes) {
  return *reuse_memory(bytes, d_avg_aterm_correction);
}

/*
 *  Memory management per stream
 *      Maintain multiple memory objects per structure
 *      Automatically increases the internal vectors for these objects
 */
template <typename T>
T* InstanceCUDA::reuse_memory(std::vector<std::unique_ptr<T>>& memories,
                              unsigned int id, uint64_t size) {
  T* ptr = NULL;

  if (memories.size() <= id) {
    ptr = new T(size);

    memories.push_back(std::unique_ptr<T>(ptr));
  } else {
    ptr = memories[id].get();
  }

  ptr->resize(size);

  return ptr;
}

cu::DeviceMemory& InstanceCUDA::allocate_device_visibilities(unsigned int id,
                                                             size_t bytes) {
  return *reuse_memory(d_visibilities_, id, bytes);
}

cu::DeviceMemory& InstanceCUDA::allocate_device_uvw(unsigned int id,
                                                    size_t bytes) {
  return *reuse_memory(d_uvw_, id, bytes);
}

cu::DeviceMemory& InstanceCUDA::allocate_device_subgrids(unsigned int id,
                                                         size_t bytes) {
  return *reuse_memory(d_subgrids_, id, bytes);
}

cu::DeviceMemory& InstanceCUDA::allocate_device_metadata(unsigned int id,
                                                         size_t bytes) {
  return *reuse_memory(d_metadata_, id, bytes);
}

/*
 * Memory management for misc device buffers
 *      Rather than storing these buffers by name,
 *      the caller gets an id that is also used to retrieve
 *      the memory from the d_misc_ vector
 */
unsigned int InstanceCUDA::allocate_device_memory(size_t bytes) {
  cu::DeviceMemory* d_misc = new cu::DeviceMemory(bytes);
  d_misc_.push_back(std::unique_ptr<cu::DeviceMemory>(d_misc));
  return d_misc_.size() - 1;
}

cu::DeviceMemory& InstanceCUDA::retrieve_device_memory(unsigned int id) {
  return *d_misc_[id];
}

/*
 * Memory management for misc page-locked host buffers
 *      Page-locking arbitrary buffers is potentially very dangerous
 *      as buffers may (partially) overlap, which will result in CUDA
 *      errors. This mechanism should only be used to register buffers
 *      that are guaranteed to be distinct and have a lifetime longer
 *      than the current InstanceCUDA object.
 */
void InstanceCUDA::register_host_memory(void* ptr, size_t bytes) {
  for (auto& memory : h_registered_) {
    if (memory->ptr() == ptr && memory->size() == bytes) {
      return;
    }
  }
  cu::RegisteredMemory* h_registered = new cu::RegisteredMemory(ptr, bytes);
  h_registered_.push_back(std::unique_ptr<cu::RegisteredMemory>(h_registered));
}

/*
 * Host memory destructor
 */
void InstanceCUDA::free_host_memory() {
  h_visibilities.reset();
  h_uvw.reset();
  h_subgrids.reset();
  h_registered_.clear();
}

/*
 * Device memory destructor
 */
void InstanceCUDA::free_device_memory() {
  d_visibilities_.clear();
  d_uvw_.clear();
  d_metadata_.clear();
  d_subgrids_.clear();
  d_misc_.clear();
  d_aterms.reset();
  d_aterms_indices.reset();
  d_aterms_derivatives.reset();
  d_avg_aterm_correction.reset();
  d_wavenumbers.reset();
  d_spheroidal.reset();
  d_grid.reset();
}

/*
 * FFT plan destructor
 */
void InstanceCUDA::free_fft_plans() {
  m_fft_grid_size = 0;
  m_fft_plan_grid.reset();
  m_fft_subgrid_bulk = 0;
  m_fft_subgrid_size = 0;
  m_fft_plan_subgrid.reset();
  d_fft_subgrid.reset();
}

/*
 * Reset device
 */
void InstanceCUDA::reset() {
  executestream.reset();
  htodstream.reset();
  dtohstream.reset();
  context->reset();
  context.reset(new cu::Context(*device));
  context->setCurrent();
  executestream.reset(new cu::Stream());
  htodstream.reset(new cu::Stream());
  dtohstream.reset(new cu::Stream());
}

void InstanceCUDA::print_device_memory_info() {
#if defined(DEBUG)
  std::cout << "InstanceCUDA::" << __func__ << std::endl;
#endif
  auto memory_total =
      device->get_total_memory() / ((float)1024 * 1024 * 1024);  // GBytes
  auto memory_free =
      device->get_free_memory() / ((float)1024 * 1024 * 1024);  // GBytes
  auto memory_used = memory_total - memory_free;
  std::clog << "Device memory -> ";
  std::clog << "total: " << memory_total << " Gb, ";
  std::clog << "used: " << memory_used << " Gb, ";
  std::clog << "free: " << memory_free << " Gb" << std::endl;
}

}  // end namespace cuda
}  // end namespace kernel
}  // end namespace idg
