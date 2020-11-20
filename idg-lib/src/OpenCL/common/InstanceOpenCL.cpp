// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <clFFT.h>

#include "InstanceOpenCL.h"
#include "PowerRecord.h"

using namespace idg::kernel::opencl;
using namespace powersensor;

#define NR_CORRELATIONS 4

namespace idg {
namespace kernel {
namespace opencl {

// Constructor
InstanceOpenCL::InstanceOpenCL(cl::Context &context, int device_nr,
                               int device_id)
    : mContext(context), mPrograms(5) {
#if defined(DEBUG)
  std::cout << __func__ << std::endl;
#endif

  // Initialize members
  device = new cl::Device(context.getInfo<CL_CONTEXT_DEVICES>()[device_id]);
  executequeue = new cl::CommandQueue(context, *device);
  htodqueue = new cl::CommandQueue(context, *device);
  dtohqueue = new cl::CommandQueue(context, *device);
  d_grid = NULL;
  d_wavenumbers = NULL;
  d_aterms = NULL;
  d_spheroidal = NULL;
  h_grid = NULL;
  h_visibilities = NULL;
  h_uvw = NULL;

  // Set kernel parameters
  set_parameters();

// Compile kernels
#pragma omp parallel for
  for (unsigned i = 0; i < 5; i++) {
    // The gridder and degridder kernels are recompiled
    // when the number of channels changes.
    const unsigned default_nr_channels = 8;
    if (i == 0) {
      compile_kernel_gridder(default_nr_channels);
    }
    if (i == 1) {
      compile_kernel_degridder(default_nr_channels);
    }
    // The adder, splitter and scaler kernel are compiled
    // only when this InstanceOpenCL is created
    if (i == 2) {
      kernel_adder = compile_kernel(2, file_adder, name_adder);
    }
    if (i == 3) {
      kernel_splitter = compile_kernel(3, file_splitter, name_splitter);
    }
    if (i == 4) {
      kernel_scaler = compile_kernel(4, file_scaler, name_scaler);
    }
  }

  // Initialize power sensor
  powerSensor = get_power_sensor(sensor_device, device_nr);

  // Kernel specific initialization
  fft_planned = false;
}

// Destructor
InstanceOpenCL::~InstanceOpenCL() {
  delete device;
  delete executequeue;
  delete htodqueue;
  delete dtohqueue;
  if (fft_planned) {
    clfftDestroyPlan(&fft_plan);
  }
  for (cl::Program *program : mPrograms) {
    delete program;
  }
  delete kernel_gridder;
  delete kernel_degridder;
  delete kernel_adder;
  delete kernel_splitter;
  delete kernel_scaler;
  delete d_grid;
  delete d_wavenumbers;
  delete d_aterms;
  delete d_spheroidal;
  delete h_grid;
  delete h_visibilities;
  delete h_uvw;
}

void InstanceOpenCL::set_parameters_default() {
  batch_gridder = 32;
  batch_degridder = 128;
  block_gridder = cl::NDRange(256, 1);
  block_degridder = cl::NDRange(256, 1);
  block_adder = cl::NDRange(128, 1);
  block_splitter = cl::NDRange(128, 1);
  block_scaler = cl::NDRange(128, 1);
}

void InstanceOpenCL::set_parameters_vega() {
  block_degridder = cl::NDRange(256, 1);
}

void InstanceOpenCL::set_parameters_fiji() {
  // Fiji parameters are default
}

void InstanceOpenCL::set_parameters_hawaii() {
  // TODO
}

void InstanceOpenCL::set_parameters_tahiti() {
  // TODO
}

void InstanceOpenCL::set_parameters() {
#if defined(DEBUG)
  std::cout << __func__ << std::endl;
#endif

  set_parameters_default();

  // Get device name
  std::string name = device->getInfo<CL_DEVICE_NAME>();

  // Overide architecture specific parameters
  if (name.compare("gfx900") == 0) {  // vega
    set_parameters_vega();
  } else if (name.compare("Fiji") == 0) {
    set_parameters_fiji();
  } else if (name.compare("Hawaii") == 0) {
    set_parameters_hawaii();
  } else if (name.compare("Tahiti") == 0) {
    set_parameters_tahiti();
  }

  // Override parameters from environment
  char *cstr_batch_size = getenv("BATCHSIZE");
  if (cstr_batch_size) {
    auto batch_size = atoi(cstr_batch_size);
    batch_gridder = batch_size;
    batch_degridder = batch_size;
  }
  char *cstr_block_size = getenv("BLOCKSIZE");
  if (cstr_block_size) {
    auto block_size = atoi(cstr_block_size);
    block_gridder = cl::NDRange(block_size, 1);
    block_degridder = cl::NDRange(block_size, 1);
  }
}

std::string InstanceOpenCL::get_compiler_flags() {
  // Parameter flags
  std::stringstream flags_constants;
  flags_constants << " -DNR_POLARIZATIONS=" << NR_CORRELATIONS;

  // OpenCL specific flags
  std::stringstream flags_opencl;
  flags_opencl << "-cl-fast-relaxed-math";
  // flags_opencl << " -save-temps";

  // OpenCL 2.0 specific flags
  float opencl_version = get_opencl_version(*device);
  if (opencl_version >= 2.0) {
    // flags_opencl << " -cl-std=CL2.0";
    // flags_opencl << " -DUSE_ATOMIC_FETCH_ADD";
  }

  // Combine flags
  std::string flags = flags_opencl.str() + flags_constants.str();

  return flags;
}

cl::Kernel *InstanceOpenCL::compile_kernel(int kernel_id, std::string file_name,
                                           std::string kernel_name,
                                           std::string flags_misc) {
#if defined(DEBUG)
  cout << __func__ << endl;
#endif

  // Source directory
  std::string srcdir = auxiliary::get_lib_dir() + "/idg-opencl";

#if defined(DEBUG)
  std::cout << "Searching for source files in: " << srcdir << std::endl;
#endif

  // Get compile flags
  std::stringstream flags_;
  flags_ << get_compiler_flags();
  flags_ << flags_misc;
  std::string flags = flags_.str();

  // Create vector of devices
  std::vector<cl::Device> devices;
  devices.push_back(*device);

  // All helper files to include in build
  std::vector<std::string> helper_files;
  helper_files.push_back("types.cl");
  helper_files.push_back("math.cl");

  // Store helper files in string
  std::stringstream source_helper_;

  for (unsigned i = 0; i < helper_files.size(); i++) {
    // Get source filename
    std::stringstream source_file_name_;
    source_file_name_ << srcdir << "/" << helper_files[i];
    std::string source_file_name = source_file_name_.str();

    // Read source from file
    std::ifstream source_file(source_file_name.c_str());
    std::string source(std::istreambuf_iterator<char>(source_file),
                       (std::istreambuf_iterator<char>()));
    source_file.close();

    // Update source helper stream
    source_helper_ << source;
  }

  std::string source_helper = source_helper_.str();

  // Get source filename
  std::stringstream source_file_name_;
  source_file_name_ << srcdir << "/" << file_name;
  std::string source_file_name = source_file_name_.str();

  // Read kernel source from file
  std::ifstream source_file(source_file_name.c_str());
  std::string source_kernel(std::istreambuf_iterator<char>(source_file),
                            (std::istreambuf_iterator<char>()));
  source_file.close();

  // Construct full source file
  std::stringstream full_source;
  full_source << source_helper;
  full_source << source_kernel;

// Print information about compilation
#if defined(DEBUG)
  std::cout << "Compiling: " << source_file_name << " " << flags << std::endl;
#endif

  // Create OpenCL program
  mPrograms[kernel_id] = new cl::Program(mContext, full_source.str());
  try {
    // Build the program
    mPrograms[kernel_id]->build(devices, flags.c_str());
  } catch (cl::Error &error) {
    std::cerr << "Compilation failed: " << error.what() << std::endl;
    std::string msg;
    mPrograms[kernel_id]->getBuildInfo(*device, CL_PROGRAM_BUILD_LOG, &msg);
    std::cout << msg << std::endl;
    exit(EXIT_FAILURE);
  }

  // Create OpenCL kernel
  cl::Kernel *kernel;
  try {
    kernel = new cl::Kernel(*(mPrograms[kernel_id]), kernel_name.c_str());
  } catch (cl::Error &error) {
    std::cerr << "Loading kernel \"" << kernel_name
              << "\" failed: " << error.what() << std::endl;
    exit(EXIT_FAILURE);
  }

  return kernel;
}  // end compile_kernel

std::ostream &operator<<(std::ostream &os, InstanceOpenCL &di) {
  cl::Device d = di.get_device();

  os << "Device: " << d.getInfo<CL_DEVICE_NAME>() << std::endl;
  os << "Driver version  : " << d.getInfo<CL_DRIVER_VERSION>() << std::endl;
  os << "Device version  : " << d.getInfo<CL_DEVICE_VERSION>() << std::endl;
  os << "Compute units   : " << d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
     << std::endl;
  os << "Clock frequency : " << d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>()
     << " MHz" << std::endl;
  os << "Global memory   : " << d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() * 1e-9
     << " Gb" << std::endl;
  os << "Local memory    : " << d.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() * 1e-6
     << " Mb" << std::endl;
  os << std::endl;

  return os;
}

/*
 * Performance measurements
 */
State InstanceOpenCL::measure() { return powerSensor->read(); }

void InstanceOpenCL::measure(PowerRecord &record, cl::CommandQueue &queue) {
  record.sensor = powerSensor;
  record.enqueue(queue);
}

typedef struct {
  PowerRecord *start;  // takes first measurement
  PowerRecord *end;    // takes second measurement
  cl::Event *event;    // used to update the Report
  Report *report;

  // Report member function pointer, used to select
  // which part of the report to update with start and
  // end when the callback for the update event is triggered.
  void (Report::*update_report)(State &, State &);
} UpdateData;

UpdateData *get_update_data(PowerSensor *sensor, Report *report,
                            void (Report::*update_report)(State &, State &)) {
  UpdateData *data = new UpdateData();
  data->start = new PowerRecord(sensor);
  data->end = new PowerRecord(sensor);
  data->event = new cl::Event();
  data->report = report;
  data->update_report = update_report;
  return data;
}

void update_report_callback(cl_event, cl_int, void *userData) {
  UpdateData *data = static_cast<UpdateData *>(userData);
  PowerRecord *start = data->start;
  PowerRecord *end = data->end;
  Report *report = data->report;
  (report->*data->update_report)(start->state, end->state);
  delete data->start;
  delete data->end;
  delete data->event;
  delete data;
}

void InstanceOpenCL::start_measurement(void *ptr) {
  UpdateData *data = (UpdateData *)ptr;

  // Schedule the first measurement (prior to kernel execution)
  data->start->enqueue(*executequeue);
}

void InstanceOpenCL::end_measurement(void *ptr) {
  UpdateData *data = (UpdateData *)ptr;

  // Schedule the second measurement (after the kernel execution)
  data->end->enqueue(*executequeue);

  // Afterwards, update the report according to the two measurements
  executequeue->enqueueMarkerWithWaitList(NULL, data->event);
  data->event->setCallback(CL_RUNNING, &update_report_callback, data);
}

typedef struct {
  int nr_timesteps;
  int nr_subgrids;
  Report *report;
  cl::Event *event;
} ReportData;

ReportData *get_report_data(int nr_timesteps, int nr_subgrids, Report *report) {
  ReportData *data = new ReportData();
  data->nr_timesteps = nr_timesteps;
  data->nr_subgrids = nr_subgrids;
  data->report = report;
  data->event = new cl::Event();
  return data;
}

void report_job_callback(cl_event, cl_int, void *userData) {
  ReportData *data = static_cast<ReportData *>(userData);
  int nr_timesteps = data->nr_timesteps;
  int nr_subgrids = data->nr_subgrids;
  Report *report = data->report;
  report->print(nr_timesteps, nr_subgrids);
  delete data->event;
  delete data;
}

void InstanceOpenCL::enqueue_report(cl::CommandQueue &queue, int nr_timesteps,
                                    int nr_subgrids) {
  ReportData *data = get_report_data(nr_timesteps, nr_subgrids, report);
  executequeue->enqueueMarkerWithWaitList(NULL, data->event);
  data->event->setCallback(CL_RUNNING, &report_job_callback, data);
}

void InstanceOpenCL::compile_kernel_gridder(unsigned nr_channels) {
  std::stringstream flags;
  flags << " -DBATCH_SIZE=" << batch_gridder / max(1, (nr_channels / 8));
  flags << " -DBLOCK_SIZE=" << block_gridder[0];
  flags << " -DNR_CHANNELS=" << nr_channels;
  kernel_gridder = compile_kernel(0, file_gridder, name_gridder, flags.str());
  nr_channels_gridder = nr_channels;
}

void InstanceOpenCL::compile_kernel_degridder(unsigned nr_channels) {
  std::stringstream flags;
  flags << " -DBATCH_SIZE=" << batch_degridder;
  flags << " -DBLOCK_SIZE=" << block_degridder[0];
  flags << " -DNR_CHANNELS=" << nr_channels;
  kernel_degridder =
      compile_kernel(1, file_degridder, name_degridder, flags.str());
  nr_channels_degridder = nr_channels;
}

/*
    Kernels
*/
void InstanceOpenCL::launch_gridder(
    int nr_timesteps, int nr_subgrids, int grid_size, int subgrid_size,
    float image_size, float w_step, int nr_channels, int nr_stations,
    cl::Buffer &d_uvw, cl::Buffer &d_wavenumbers, cl::Buffer &d_visibilities,
    cl::Buffer &d_spheroidal, cl::Buffer &d_aterm, cl::Buffer &d_metadata,
    cl::Buffer &d_subgrid) {
  if (nr_channels_gridder != nr_channels) {
    compile_kernel_gridder(nr_channels);
  }

  int local_size_x = block_gridder[0];
  int local_size_y = block_gridder[1];
  cl::NDRange global_size(local_size_x * nr_subgrids, local_size_y);
  kernel_gridder->setArg(0, grid_size);
  kernel_gridder->setArg(1, subgrid_size);
  kernel_gridder->setArg(2, image_size);
  kernel_gridder->setArg(3, w_step);
  kernel_gridder->setArg(4, nr_stations);
  kernel_gridder->setArg(5, d_uvw);
  kernel_gridder->setArg(6, d_wavenumbers);
  kernel_gridder->setArg(7, d_visibilities);
  kernel_gridder->setArg(8, d_spheroidal);
  kernel_gridder->setArg(9, d_aterm);
  kernel_gridder->setArg(10, d_metadata);
  kernel_gridder->setArg(11, d_subgrid);
  try {
    UpdateData *data =
        get_update_data(powerSensor, report, &Report::update_gridder);
    start_measurement(data);
    executequeue->enqueueNDRangeKernel(*kernel_gridder, cl::NullRange,
                                       global_size, block_gridder);
    end_measurement(data);
  } catch (cl::Error &error) {
    std::cerr << "Error launching gridder: " << error.what() << std::endl;
    exit(EXIT_FAILURE);
  }
}

void InstanceOpenCL::launch_degridder(
    int nr_timesteps, int nr_subgrids, int grid_size, int subgrid_size,
    float image_size, float w_step, int nr_channels, int nr_stations,
    cl::Buffer &d_uvw, cl::Buffer &d_wavenumbers, cl::Buffer &d_visibilities,
    cl::Buffer &d_spheroidal, cl::Buffer &d_aterm, cl::Buffer &d_metadata,
    cl::Buffer &d_subgrid) {
  if (nr_channels_degridder != nr_channels) {
    compile_kernel_degridder(nr_channels);
  }

  int local_size_x = block_degridder[0];
  int local_size_y = block_degridder[1];
  cl::NDRange global_size(local_size_x * nr_subgrids, local_size_y);
  kernel_degridder->setArg(0, grid_size);
  kernel_degridder->setArg(1, subgrid_size);
  kernel_degridder->setArg(2, image_size);
  kernel_degridder->setArg(3, w_step);
  kernel_degridder->setArg(4, nr_stations);
  kernel_degridder->setArg(5, d_uvw);
  kernel_degridder->setArg(6, d_wavenumbers);
  kernel_degridder->setArg(7, d_visibilities);
  kernel_degridder->setArg(8, d_spheroidal);
  kernel_degridder->setArg(9, d_aterm);
  kernel_degridder->setArg(10, d_metadata);
  kernel_degridder->setArg(11, d_subgrid);
  try {
    UpdateData *data =
        get_update_data(powerSensor, report, &Report::update_degridder);
    start_measurement(data);
    executequeue->enqueueNDRangeKernel(*kernel_degridder, cl::NullRange,
                                       global_size, block_degridder);
    end_measurement(data);
  } catch (cl::Error &error) {
    std::cerr << "Error launching degridder: " << error.what() << std::endl;
    exit(EXIT_FAILURE);
  }
}

void InstanceOpenCL::plan_fft(unsigned size, unsigned batch) {
  // Check wheter a new plan has to be created
  if (!fft_planned || size != fft_planned_size || batch != fft_planned_batch) {
    // Destroy old plan (if any)
    if (fft_planned) {
      clfftDestroyPlan(&fft_plan);
    }

    // Create new plan
    size_t lengths[2] = {(size_t)size, (size_t)size};
    clfftCreateDefaultPlan(&fft_plan, mContext(), CLFFT_2D, lengths);

    // Set plan parameters
    clfftSetPlanPrecision(fft_plan, CLFFT_SINGLE);
    clfftSetLayout(fft_plan, CLFFT_COMPLEX_INTERLEAVED,
                   CLFFT_COMPLEX_INTERLEAVED);
    clfftSetResultLocation(fft_plan, CLFFT_INPLACE);
    int distance = size * size;
    clfftSetPlanDistance(fft_plan, distance, distance);
    clfftSetPlanBatchSize(fft_plan, batch * NR_CORRELATIONS);

    // Update parameters
    fft_planned_size = size;
    fft_planned_batch = batch;

    // Bake plan
    cl_command_queue *queue = &(*executequeue)();
    clfftStatus status = clfftBakePlan(fft_plan, 1, queue, NULL, NULL);
    if (status != CL_SUCCESS) {
      std::cerr << "Error baking fft plan" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  fft_planned = true;
}

void InstanceOpenCL::launch_fft(cl::Buffer &d_data,
                                DomainAtoDomainB direction) {
  clfftDirection sign = (direction == FourierDomainToImageDomain)
                            ? CLFFT_BACKWARD
                            : CLFFT_FORWARD;
  size_t batch;
  clfftGetPlanBatchSize(fft_plan, &batch);
  UpdateData *data =
      batch > NR_CORRELATIONS
          ? get_update_data(powerSensor, report, &Report::update_subgrid_fft)
          : get_update_data(powerSensor, report, &Report::update_grid_fft);
  start_measurement(data);
  cl_command_queue *queue = &(*executequeue)();
  clfftStatus status = clfftEnqueueTransform(fft_plan, sign, 1, queue, 0, NULL,
                                             NULL, &d_data(), NULL, NULL);
  end_measurement(data);
  if (status != CL_SUCCESS) {
    std::cerr << "Error enqueing fft plan" << std::endl;
    exit(EXIT_FAILURE);
  }
}

void InstanceOpenCL::launch_adder(int nr_subgrids, int grid_size,
                                  int subgrid_size, cl::Buffer &d_metadata,
                                  cl::Buffer &d_subgrid, cl::Buffer &d_grid) {
  int local_size_x = block_adder[0];
  int local_size_y = block_adder[1];
  cl::NDRange global_size(local_size_x * nr_subgrids, local_size_y);
  kernel_adder->setArg(0, grid_size);
  kernel_adder->setArg(1, subgrid_size);
  kernel_adder->setArg(2, d_metadata);
  kernel_adder->setArg(3, d_subgrid);
  kernel_adder->setArg(4, d_grid);
  try {
    UpdateData *data =
        get_update_data(powerSensor, report, &Report::update_adder);
    start_measurement(data);
    executequeue->enqueueNDRangeKernel(*kernel_adder, cl::NullRange,
                                       global_size, block_adder);
    end_measurement(data);
  } catch (cl::Error &error) {
    std::cerr << "Error launching adder: " << error.what() << std::endl;
    exit(EXIT_FAILURE);
  }
}

void InstanceOpenCL::launch_splitter(int nr_subgrids, int grid_size,
                                     int subgrid_size, cl::Buffer &d_metadata,
                                     cl::Buffer &d_subgrid,
                                     cl::Buffer &d_grid) {
  int local_size_x = block_splitter[0];
  int local_size_y = block_splitter[1];
  cl::NDRange global_size(local_size_x * nr_subgrids, local_size_y);
  kernel_splitter->setArg(0, grid_size);
  kernel_splitter->setArg(1, subgrid_size);
  kernel_splitter->setArg(2, d_metadata);
  kernel_splitter->setArg(3, d_subgrid);
  kernel_splitter->setArg(4, d_grid);
  try {
    UpdateData *data =
        get_update_data(powerSensor, report, &Report::update_splitter);
    start_measurement(data);
    executequeue->enqueueNDRangeKernel(*kernel_splitter, cl::NullRange,
                                       global_size, block_splitter);
    end_measurement(data);
  } catch (cl::Error &error) {
    std::cerr << "Error launching splitter: " << error.what() << std::endl;
    exit(EXIT_FAILURE);
  }
}

void InstanceOpenCL::launch_scaler(int nr_subgrids, int subgrid_size,
                                   cl::Buffer &d_subgrid) {
  int local_size_x = block_scaler[0];
  int local_size_y = block_scaler[1];
  cl::NDRange global_size(local_size_x * nr_subgrids, local_size_y);
  kernel_scaler->setArg(0, subgrid_size);
  kernel_scaler->setArg(1, d_subgrid);
  try {
    UpdateData *data =
        get_update_data(powerSensor, report, &Report::update_scaler);
    start_measurement(data);
    executequeue->enqueueNDRangeKernel(*kernel_scaler, cl::NullRange,
                                       global_size, block_scaler);
    end_measurement(data);
  } catch (cl::Error &error) {
    std::cerr << "Error launching scaler: " << error.what() << std::endl;
    exit(EXIT_FAILURE);
  }
}

cl::Buffer *InstanceOpenCL::reuse_memory(uint64_t size, cl::Buffer *buffer,
                                         cl_mem_flags flags, void *ptr) {
  if (buffer && size != buffer->getInfo<CL_MEM_SIZE>()) {
    delete buffer;
    buffer = new cl::Buffer(mContext, flags, size, ptr);
  } else if (!buffer) {
    buffer = new cl::Buffer(mContext, flags, size, ptr);
  }
  return buffer;
}

cl::Buffer &InstanceOpenCL::get_device_grid(unsigned int grid_size) {
  if (grid_size > 0) {
    auto size = auxiliary::sizeof_grid(grid_size);
    d_grid = reuse_memory(size, d_grid, CL_MEM_READ_WRITE);
  }
  return *d_grid;
}

cl::Buffer &InstanceOpenCL::get_device_wavenumbers(unsigned int nr_channels) {
  if (nr_channels > 0) {
    auto size = auxiliary::sizeof_wavenumbers(nr_channels);
    d_wavenumbers = reuse_memory(size, d_wavenumbers, CL_MEM_READ_WRITE);
  }
  return *d_wavenumbers;
}

cl::Buffer &InstanceOpenCL::get_device_aterms(unsigned int nr_stations,
                                              unsigned int nr_timeslots,
                                              unsigned int subgrid_size) {
  if (nr_stations > 0 && nr_timeslots > 0 && subgrid_size > 0) {
    auto size =
        auxiliary::sizeof_aterms(nr_stations, nr_timeslots, subgrid_size);
    d_aterms = reuse_memory(size, d_aterms, CL_MEM_READ_WRITE);
  }
  return *d_aterms;
}

cl::Buffer &InstanceOpenCL::get_device_spheroidal(unsigned int subgrid_size) {
  if (subgrid_size > 0) {
    auto size = auxiliary::sizeof_spheroidal(subgrid_size);
    d_spheroidal = reuse_memory(size, d_spheroidal, CL_MEM_READ_WRITE);
  }
  return *d_spheroidal;
}

cl::Buffer &InstanceOpenCL::get_host_grid(unsigned int grid_size) {
  if (grid_size > 0) {
    auto size = auxiliary::sizeof_grid(grid_size);
    h_grid = reuse_memory(size, h_grid, CL_MEM_ALLOC_HOST_PTR);
  }
  return *h_grid;
}

cl::Buffer &InstanceOpenCL::get_host_visibilities(unsigned int nr_baselines,
                                                  unsigned int nr_timesteps,
                                                  unsigned int nr_channels,
                                                  void *ptr) {
  if (nr_baselines > 0 && nr_timesteps > 0 && nr_channels > 0) {
    auto size =
        auxiliary::sizeof_visibilities(nr_baselines, nr_timesteps, nr_channels);
    cl_mem_flags mem_flags =
        (ptr != NULL) ? CL_MEM_USE_HOST_PTR : CL_MEM_ALLOC_HOST_PTR;
    h_visibilities = reuse_memory(size, h_visibilities, mem_flags, ptr);
  }
  return *h_visibilities;
}

cl::Buffer &InstanceOpenCL::get_host_uvw(unsigned int nr_baselines,
                                         unsigned int nr_timesteps, void *ptr) {
  if (nr_baselines > 0 && nr_timesteps > 0) {
    auto size = auxiliary::sizeof_uvw(nr_baselines, nr_timesteps);
    cl_mem_flags mem_flags =
        (ptr != NULL) ? CL_MEM_USE_HOST_PTR : CL_MEM_ALLOC_HOST_PTR;
    h_uvw = reuse_memory(size, h_uvw, mem_flags, ptr);
  }
  return *h_uvw;
}

}  // end namespace opencl
}  // end namespace kernel
}  // end namespace idg
