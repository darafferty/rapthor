// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <string>

#include <cuda.h>
#include <cudaProfiler.h>

#include "CUDA.h"

#include "InstanceCUDA.h"

//#define DEBUG_COMPUTE_JOBSIZE

using namespace idg::kernel::cuda;

namespace idg {
namespace proxy {
namespace cuda {
CUDA::CUDA(ProxyInfo info) : mInfo(info) {
#if defined(DEBUG)
  std::cout << "CUDA::" << __func__ << std::endl;
#endif

  cu::init();
  init_devices();
  print_devices();
  print_compiler_flags();
  cuProfilerStart();
};

CUDA::~CUDA() {
  cuProfilerStop();
  free_devices();
}

void CUDA::init_devices() {
  // Get list of all device numbers
  char* char_cuda_device = getenv("CUDA_DEVICE");
  std::vector<int> device_numbers;
  if (!char_cuda_device) {
    // Use device 0 if no CUDA devices were specified
    device_numbers.push_back(0);
  } else {
    device_numbers = idg::auxiliary::split_int(char_cuda_device, ",");
  }

  // Create a device instance for every device
  for (unsigned i = 0; i < device_numbers.size(); i++) {
    InstanceCUDA* device = new InstanceCUDA(mInfo, i, device_numbers[i]);
    devices.push_back(device);
  }
}

void CUDA::free_devices() {
  for (InstanceCUDA* device : devices) {
    delete device;
  }
}

void CUDA::print_devices() {
  std::cout << "Devices: " << std::endl;
  for (InstanceCUDA* device : devices) {
    std::cout << *device;
  }
  std::cout << std::endl;
}

void CUDA::print_compiler_flags() {
  std::cout << "Compiler flags: " << std::endl;
  for (InstanceCUDA* device : devices) {
    std::cout << device->get_compiler_flags() << std::endl;
  }
  std::cout << std::endl;
}

unsigned int CUDA::get_num_devices() const { return devices.size(); }

InstanceCUDA& CUDA::get_device(unsigned int i) const { return *(devices[i]); }

ProxyInfo CUDA::default_info() {
#if defined(DEBUG)
  std::cout << "CUDA::" << __func__ << std::endl;
#endif

  std::string srcdir = auxiliary::get_lib_dir() + "/idg-cuda";

#if defined(DEBUG)
  std::cout << "Searching for source files in: " << srcdir << std::endl;
#endif

  // Create temp directory
  char _tmpdir[] = "/tmp/idg-XXXXXX";
  char* tmpdir = mkdtemp(_tmpdir);
#if defined(DEBUG)
  std::cout << "Temporary files will be stored in: " << tmpdir << std::endl;
#endif

  // Create proxy info
  ProxyInfo p;
  p.set_path_to_src(srcdir);
  p.set_path_to_lib(tmpdir);

  return p;
}  // end default_info

std::unique_ptr<auxiliary::Memory> CUDA::allocate_memory(size_t bytes) {
  return std::unique_ptr<auxiliary::Memory>(new cu::HostMemory(bytes));
}

std::vector<int> CUDA::compute_jobsize(const Plan& plan,
                                       const unsigned int nr_stations,
                                       const unsigned int nr_timeslots,
                                       const unsigned int nr_timesteps,
                                       const unsigned int nr_channels,
                                       const unsigned int subgrid_size) {
#if defined(DEBUG)
  std::cout << "CUDA::" << __func__ << std::endl;
#endif

  // Get additional parameters
  unsigned int nr_baselines = plan.get_nr_baselines();

  // Check if parameters have changed
  bool reset = false;
  if (nr_stations != m_gridding_state.nr_stations) {
    reset = true;
  };
  if (nr_timeslots != m_gridding_state.nr_timeslots) {
    reset = true;
  };
  if (nr_timesteps != m_gridding_state.nr_timesteps) {
    reset = true;
  };
  if (nr_channels != m_gridding_state.nr_channels) {
    reset = true;
  };
  if (subgrid_size != m_gridding_state.subgrid_size) {
    reset = true;
  };

  for (unsigned i = 0; i < m_gridding_state.jobsize.size(); i++) {
    unsigned int jobsize = m_gridding_state.jobsize[i];
    unsigned int nr_subgrids =
        plan.get_max_nr_subgrids(0, nr_baselines, jobsize);
    unsigned int max_nr_subgrids = m_gridding_state.max_nr_subgrids[i];
    if (nr_subgrids > max_nr_subgrids) {
      reset = true;
    };
  }

  // Reuse same jobsize if no parameters have changed
  if (!reset) {
#if defined(DEBUG_COMPUTE_JOBSIZE)
    std::clog << "Reuse previous jobsize" << std::endl;
#endif
    return m_gridding_state.jobsize;
  } else {
    // Free all the memory allocated by initialize
    // such that the new jobsize can be properly computed
    cleanup();
  }

  // Set parameters
  m_gridding_state.nr_stations = nr_stations;
  m_gridding_state.nr_timeslots = nr_timeslots;
  m_gridding_state.nr_timesteps = nr_timesteps;
  m_gridding_state.nr_channels = nr_channels;
  m_gridding_state.subgrid_size = subgrid_size;
  m_gridding_state.nr_baselines = nr_baselines;

// Print parameters
#if defined(DEBUG_COMPUTE_JOBSIZE)
  std::cout << "nr_stations  = " << nr_stations << std::endl;
  std::cout << "nr_timeslots = " << nr_timeslots << std::endl;
  std::cout << "nr_timesteps = " << nr_timesteps << std::endl;
  std::cout << "nr_channels  = " << nr_channels << std::endl;
  std::cout << "subgrid_size = " << subgrid_size << std::endl;
  std::cout << "nr_baselines = " << nr_baselines << std::endl;
#endif

  // Read maximum jobsize from environment
  char* cstr_max_jobsize = getenv("MAX_JOBSIZE");
  auto max_jobsize = cstr_max_jobsize ? atoi(cstr_max_jobsize) : 0;
#if defined(DEBUG_COMPUTE_JOBSIZE)
  std::cout << "max_jobsize  = " << max_jobsize << std::endl;
#endif

  // Compute the maximum number of subgrids for any baseline
  int max_nr_subgrids_bl = plan.get_max_nr_subgrids();

  // Compute the amount of bytes needed for that job
  size_t bytes_job = 0;
  bytes_job += auxiliary::sizeof_visibilities(1, nr_timesteps, nr_channels);
  bytes_job += auxiliary::sizeof_uvw(1, nr_timesteps);
  bytes_job += auxiliary::sizeof_subgrids(max_nr_subgrids_bl, subgrid_size);
  bytes_job += auxiliary::sizeof_metadata(max_nr_subgrids_bl);
  bytes_job *= m_max_nr_streams;

  // Compute the amount of memory needed for data that is identical for all jobs
  size_t bytes_static = 0;
  bytes_static +=
      auxiliary::sizeof_aterms(nr_stations, nr_timeslots, subgrid_size);
  bytes_static += auxiliary::sizeof_spheroidal(subgrid_size);
  bytes_static += auxiliary::sizeof_aterms_indices(nr_baselines, nr_timesteps);
  bytes_static += auxiliary::sizeof_wavenumbers(nr_channels);
  bytes_static += auxiliary::sizeof_avg_aterm_correction(subgrid_size);

// Print amount of bytes required
#if defined(DEBUG_COMPUTE_JOBSIZE)
  std::clog << "Bytes required for static data: " << bytes_static << std::endl;
  std::clog << "Bytes required for job data: " << bytes_job << std::endl;
#endif

  // Adjust jobsize to amount of available device memory
  unsigned nr_devices = devices.size();
  std::vector<int> jobsize(nr_devices);
  std::vector<int> max_nr_subgrids_job(nr_devices);
  for (unsigned i = 0; i < nr_devices; i++) {
    InstanceCUDA* device = devices[i];
    cu::Context& context = device->get_context();
    context.setCurrent();

    // Print device number
    if (nr_devices > 1) {
#if defined(DEBUG_COMPUTE_JOBSIZE)
      std::clog << "GPU " << i << ", ";
#endif
    }

    // Get amount of memory available on device
    auto bytes_free = device->get_device().get_free_memory();
#if defined(DEBUG_COMPUTE_JOBSIZE)
    std::clog << "Bytes free: " << bytes_free << std::endl;
#endif

    // Print reserved memory
    if (m_fraction_reserved > 0) {
#if defined(DEBUG_COMPUTE_JOBSIZE)
      std::clog << "Bytes reserved: "
                << (long)(bytes_free * m_fraction_reserved) << std::endl;
#endif
    }

    // Check whether the static data and job data fits at all
    if (bytes_free < (bytes_static + bytes_job)) {
      std::cerr << "Error! Not enough (free) memory on device to continue.";
      std::cerr << std::endl;
      exit(EXIT_FAILURE);
    }

    // Subtract the space for static memory from the amount of free memory
    bytes_free -= bytes_static;

    // Compute jobsize
    jobsize[i] = (bytes_free * (1 - m_fraction_reserved)) / bytes_job;
    jobsize[i] = max_jobsize > 0 ? min(jobsize[i], max_jobsize) : jobsize[i];
    jobsize[i] = min(jobsize[i], nr_baselines);

    // Sanity check
    if (jobsize[i] == 0) {
      std::stringstream message;
      message << std::setprecision(1);
      message << "jobsize == 0" << std::endl;
      message << "GPU memory required for static data: " << bytes_static * 1e-6
              << " Mb" << std::endl;
      message << "GPU memory required for job data: " << bytes_job * 1e-6
              << " Mb" << std::endl;
      message << "GPU free memory: " << int(bytes_free * 1e-6) << " Mb ("
              << int(bytes_free * m_fraction_reserved * 1e-6)
              << " Mb reserved)";
      throw std::runtime_error(message.str());
    }

// Print jobsize
#if defined(DEBUG_COMPUTE_JOBSIZE)
    printf("Jobsize: %d\n", jobsize[i]);
#endif

    // Get maximum number of subgrids for any job
    max_nr_subgrids_job[i] =
        plan.get_max_nr_subgrids(0, nr_baselines, jobsize[i]);
  }

  m_gridding_state.jobsize = jobsize;
  m_gridding_state.max_nr_subgrids = max_nr_subgrids_job;

  return jobsize;
}  // end compute_jobsize

void CUDA::initialize(
    const Plan& plan, const float w_step, const Array1D<float>& shift,
    const float cell_size, const unsigned int kernel_size,
    const unsigned int subgrid_size, const Array1D<float>& frequencies,
    const Array3D<Visibility<std::complex<float>>>& visibilities,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterms_offsets,
    const Array2D<float>& spheroidal) {
#if defined(DEBUG)
  std::cout << "CUDA::" << __func__ << std::endl;
#endif

  cu::Marker marker("initialize");
  marker.start();

  // Arguments
  auto nr_channels = frequencies.get_x_dim();
  auto nr_stations = aterms.get_z_dim();
  auto nr_timeslots = aterms.get_w_dim();
  auto nr_baselines = visibilities.get_z_dim();
  auto nr_timesteps = visibilities.get_y_dim();

  // Convert frequencies to wavenumbers
  Array1D<float> wavenumbers = compute_wavenumbers(frequencies);

  // Compute jobsize
  compute_jobsize(plan, nr_stations, nr_timeslots, nr_timesteps, nr_channels,
                  subgrid_size);

  // Allocate and initialize device memory
  for (unsigned d = 0; d < get_num_devices(); d++) {
    InstanceCUDA& device = get_device(d);
    device.set_context();
    auto jobsize = m_gridding_state.jobsize[d];
    auto max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);
    cu::Stream& htodstream = device.get_htod_stream();

    // Wavenumbers
    cu::DeviceMemory& d_wavenumbers =
        device.allocate_device_wavenumbers(wavenumbers.bytes());
    htodstream.memcpyHtoDAsync(d_wavenumbers, wavenumbers.data(),
                               wavenumbers.bytes());

    // Spheroidal
    cu::DeviceMemory& d_spheroidal =
        device.allocate_device_spheroidal(spheroidal.bytes());
    htodstream.memcpyHtoDAsync(d_spheroidal, spheroidal.data(),
                               spheroidal.bytes());

    // Aterms
    cu::DeviceMemory& d_aterms = device.allocate_device_aterms(aterms.bytes());
    htodstream.memcpyHtoDAsync(d_aterms, aterms.data(), aterms.bytes());

    // Aterms indicies
    size_t sizeof_aterm_indices =
        auxiliary::sizeof_aterms_indices(nr_baselines, nr_timesteps);
    cu::DeviceMemory& d_aterms_indices =
        device.allocate_device_aterms_indices(sizeof_aterm_indices);
    htodstream.memcpyHtoDAsync(d_aterms_indices, plan.get_aterm_indices_ptr(),
                               sizeof_aterm_indices);

    // Average aterm correction
    if (m_avg_aterm_correction.size()) {
      size_t sizeof_avg_aterm_correction =
          auxiliary::sizeof_avg_aterm_correction(subgrid_size);
      cu::DeviceMemory& d_avg_aterm_correction =
          device.allocate_device_avg_aterm_correction(
              sizeof_avg_aterm_correction);
      htodstream.memcpyHtoDAsync(d_avg_aterm_correction,
                                 m_avg_aterm_correction.data(),
                                 sizeof_avg_aterm_correction);
    } else {
      device.allocate_device_avg_aterm_correction(0);
    }

    // Dynamic memory (per thread)
    for (unsigned t = 0; t < m_max_nr_streams; t++) {
      // Visibilities
      size_t sizeof_visibilities =
          auxiliary::sizeof_visibilities(jobsize, nr_timesteps, nr_channels);
      device.allocate_device_visibilities(t, sizeof_visibilities);

      // UVW coordinates
      size_t sizeof_uvw = auxiliary::sizeof_uvw(jobsize, nr_timesteps);
      device.allocate_device_uvw(t, sizeof_uvw);

      // Subgrids
      size_t sizeof_subgrids =
          auxiliary::sizeof_subgrids(max_nr_subgrids, subgrid_size);
      device.allocate_device_subgrids(t, sizeof_subgrids);

      // Metadata
      size_t sizeof_metadata = auxiliary::sizeof_metadata(max_nr_subgrids);
      device.allocate_device_metadata(t, sizeof_metadata);
    }

    // Initialize job data
    jobs.clear();
    for (unsigned bl = 0; bl < nr_baselines; bl += jobsize) {
      unsigned int first_bl, last_bl, current_nr_baselines;
      plan.initialize_job(nr_baselines, jobsize, bl, &first_bl, &last_bl,
                          &current_nr_baselines);
      if (current_nr_baselines == 0) continue;
      JobData job;
      job.current_time_offset = first_bl * nr_timesteps;
      job.current_nr_baselines = current_nr_baselines;
      job.current_nr_subgrids =
          plan.get_nr_subgrids(first_bl, current_nr_baselines);
      job.current_nr_timesteps =
          plan.get_nr_timesteps(first_bl, current_nr_baselines);
      job.metadata_ptr = (void*)plan.get_metadata_ptr(first_bl);
      job.uvw_ptr = uvw.data(first_bl, 0);
      job.visibilities_ptr = visibilities.data(first_bl, 0, 0);
      jobs.push_back(job);
    }

    // Plan subgrid fft
    device.plan_subgrid_fft(subgrid_size, max_nr_subgrids);

    // Wait for memory copies
    htodstream.synchronize();
  }

  marker.end();
}  // end initialize

void CUDA::cleanup() {
#if defined(DEBUG)
  std::cout << "CUDA::" << __func__ << std::endl;
#endif

  cu::Marker marker("cleanup");
  marker.start();

  for (unsigned d = 0; d < get_num_devices(); d++) {
    InstanceCUDA& device = get_device(d);
    device.set_context();
    device.free_device_wavenumbers();
    device.free_device_spheroidal();
    device.free_device_aterms();
    device.free_device_aterms_indices();
    device.free_device_avg_aterm_correction();
    device.free_device_visibilities();
    device.free_device_uvw();
    device.free_device_subgrids();
    device.free_device_metadata();
    device.unmap_host_memory();
    device.free_fft_plans();
  }

  marker.end();
}
}  // end namespace cuda
}  // end namespace proxy
}  // end namespace idg
