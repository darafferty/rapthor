// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <string>

#include <cuda.h>
#include <cudaProfiler.h>

#include "CUDA.h"

#include "InstanceCUDA.h"

#if defined(DEBUG)
#define DEBUG_COMPUTE_JOBSIZE
#define DEBUG_MEMORY_FRAGMENTATION
#endif

using namespace idg::kernel::cuda;

namespace idg {
namespace proxy {
namespace cuda {
CUDA::CUDA(ProxyInfo info)
    : hostPowerSensor(powersensor::get_power_sensor(powersensor::sensor_host)),
      mInfo(info) {
#if defined(DEBUG)
  std::cout << "CUDA::" << __func__ << std::endl;
#endif

  cu::init();
  init_devices();
  initialize_buffers();
  print_devices();
  print_compiler_flags();
  cuProfilerStart();
};

CUDA::~CUDA() {
  cuProfilerStop();
  free_buffers();
  m_buffers.d_grid.reset();
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
    InstanceCUDA* device = new InstanceCUDA(mInfo, device_numbers[i]);
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

void CUDA::initialize_buffers() {
#if defined(DEBUG)
  std::cout << "CUDA::" << __func__ << std::endl;
#endif

  free_buffers();

  const cu::Context& context = get_device(0).get_context();
  m_buffers.d_wavenumbers.reset(new cu::DeviceMemory(context, 0));
  m_buffers.d_spheroidal.reset(new cu::DeviceMemory(context, 0));
  m_buffers.d_aterms.reset(new cu::DeviceMemory(context, 0));
  m_buffers.d_avg_aterm.reset(new cu::DeviceMemory(context, 0));
  // d_grid is handled seperately
  // d_lmnp is handled in GenericOptimized

  for (unsigned t = 0; t < m_max_nr_streams; t++) {
    m_buffers.d_visibilities_.emplace_back(new cu::DeviceMemory(context, 0));
    m_buffers.d_uvw_.emplace_back(new cu::DeviceMemory(context, 0));
    m_buffers.d_subgrids_.emplace_back(new cu::DeviceMemory(context, 0));
    m_buffers.d_metadata_.emplace_back(new cu::DeviceMemory(context, 0));
  }

  // Only one aterms_indices buffer is used for gridding and degridding,
  // multiple buffers are only used for calibration in GenericOptimized.
  m_buffers.d_aterms_indices_.emplace_back(new cu::DeviceMemory(context, 0));

  // d_sums_ is handled in GenericOptimized

  m_buffers.h_subgrids.reset(new cu::HostMemory(context, 0));
}

void CUDA::free_buffers() {
#if defined(DEBUG)
  std::cout << "CUDA::" << __func__ << std::endl;
#endif

  m_buffers.d_wavenumbers.reset();
  m_buffers.d_spheroidal.reset();
  m_buffers.d_aterms.reset();
  m_buffers.d_avg_aterm.reset();
  // d_grid is handled seperately
  m_buffers.d_lmnp.reset();

  m_buffers.d_visibilities_.resize(0);
  m_buffers.d_uvw_.resize(0);
  m_buffers.d_subgrids_.resize(0);
  m_buffers.d_metadata_.resize(0);
  m_buffers.d_weights_.resize(0);
  m_buffers.d_aterms_indices_.resize(0);
  m_buffers.d_sums_.resize(0);

  m_buffers.h_subgrids.reset();
}

std::unique_ptr<auxiliary::Memory> CUDA::allocate_memory(size_t bytes) {
  const cu::Context& context = get_device(0).get_context();
  return std::unique_ptr<auxiliary::Memory>(new cu::HostMemory(context, bytes));
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

    // Print device number
    if (nr_devices > 1) {
#if defined(DEBUG_COMPUTE_JOBSIZE)
      std::clog << "GPU " << i << ", ";
#endif
    }

    // Get amount of memory available on device
    size_t bytes_free = device->get_free_memory();
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
    const Plan& plan, const Array1D<float>& frequencies,
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
  auto subgrid_size = plan.get_subgrid_size();
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

  try {
    // Allocate and initialize device memory
    for (unsigned d = 0; d < get_num_devices(); d++) {
      InstanceCUDA& device = get_device(d);
      auto jobsize = m_gridding_state.jobsize[d];
      auto max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);
      cu::Stream& htodstream = device.get_htod_stream();

      // Wavenumbers
      m_buffers.d_wavenumbers->resize(wavenumbers.bytes());
      htodstream.memcpyHtoDAsync(*m_buffers.d_wavenumbers, wavenumbers.data(),
                                 wavenumbers.bytes());

      // Spheroidal
      m_buffers.d_spheroidal->resize((spheroidal.bytes()));
      htodstream.memcpyHtoDAsync(*m_buffers.d_spheroidal, spheroidal.data(),
                                 spheroidal.bytes());

      // Aterms
      m_buffers.d_aterms->resize(aterms.bytes());
      htodstream.memcpyHtoDAsync(*m_buffers.d_aterms, aterms.data(),
                                 aterms.bytes());

      // Aterms indices
      size_t sizeof_aterms_indices =
          auxiliary::sizeof_aterms_indices(nr_baselines, nr_timesteps);
      m_buffers.d_aterms_indices_[0]->resize(sizeof_aterms_indices);
      htodstream.memcpyHtoDAsync(*m_buffers.d_aterms_indices_[0],
                                 plan.get_aterm_indices_ptr(),
                                 sizeof_aterms_indices);

      // Average aterm correction
      size_t sizeof_avg_aterm_correction =
          m_avg_aterm_correction.size() > 0
              ? auxiliary::sizeof_avg_aterm_correction(subgrid_size)
              : 0;
      m_buffers.d_avg_aterm->resize(sizeof_avg_aterm_correction);
      htodstream.memcpyHtoDAsync(*m_buffers.d_avg_aterm,
                                 m_avg_aterm_correction.data(),
                                 sizeof_avg_aterm_correction);

      // Dynamic memory (per thread)
      for (unsigned t = 0; t < m_max_nr_streams; t++) {
        // Visibilities
        size_t sizeof_visibilities =
            auxiliary::sizeof_visibilities(jobsize, nr_timesteps, nr_channels);
        m_buffers.d_visibilities_[t]->resize(sizeof_visibilities);

        // UVW coordinates
        size_t sizeof_uvw = auxiliary::sizeof_uvw(jobsize, nr_timesteps);
        m_buffers.d_uvw_[t]->resize(sizeof_uvw);

        // Subgrids
        size_t sizeof_subgrids =
            auxiliary::sizeof_subgrids(max_nr_subgrids, subgrid_size);
        m_buffers.d_subgrids_[t]->resize(sizeof_subgrids);

        // Metadata
        size_t sizeof_metadata = auxiliary::sizeof_metadata(max_nr_subgrids);
        m_buffers.d_metadata_[t]->resize(sizeof_metadata);
      }

      // Initialize job data
      jobs.clear();
      for (unsigned bl = 0; bl < nr_baselines; bl += jobsize) {
        unsigned int first_bl, last_bl, current_nr_baselines;
        plan.initialize_job(nr_baselines, jobsize, bl, &first_bl, &last_bl,
                            &current_nr_baselines);
        if (current_nr_baselines == 0) continue;
        JobData job;
        job.first_bl = first_bl;
        job.current_time_offset = first_bl * nr_timesteps;
        job.current_nr_baselines = current_nr_baselines;
        job.current_nr_subgrids =
            plan.get_nr_subgrids(first_bl, current_nr_baselines);
        job.current_nr_timesteps =
            plan.get_nr_timesteps(first_bl, current_nr_baselines);
        job.metadata_ptr = plan.get_metadata_ptr(first_bl);
        job.uvw_ptr = uvw.data(first_bl, 0);
        job.visibilities_ptr = visibilities.data(first_bl, 0, 0);
        jobs.push_back(job);
      }

      // Plan subgrid fft
      device.plan_subgrid_fft(subgrid_size, max_nr_subgrids);

      // Wait for memory copies
      htodstream.synchronize();

      // Remove all pre-existing events
      device.free_events();
    }
  } catch (cu::Error<CUresult>& error) {
    if (error == CUDA_ERROR_OUT_OF_MEMORY) {
      // There should be sufficient GPU memory available,
      // since compute_jobsize completed succesfully.
#if defined(DEBUG_MEMORY_FRAGMENTATION)
      std::cout << "Memory fragmentation detected, retrying to allocate device "
                   "memory."
                << std::endl;
#endif

      // Free all device memory
      for (unsigned d = 0; d < get_num_devices(); d++) {
        InstanceCUDA& device = get_device(d);
        initialize_buffers();
        device.free_fft_plans();
      }

      // Try again to allocate device memory
      initialize(plan, frequencies, visibilities, uvw, baselines, aterms,
                 aterms_offsets, spheroidal);
    } else {
      throw;
    }
  }

  marker.end();
}  // end initialize

void CUDA::cleanup() {
#if defined(DEBUG)
  std::cout << "CUDA::" << __func__ << std::endl;
#endif

  cu::Marker marker("cleanup");
  marker.start();

  initialize_buffers();

  for (unsigned d = 0; d < get_num_devices(); d++) {
    InstanceCUDA& device = get_device(d);
    device.free_fft_plans();
  }

  marker.end();
}
}  // end namespace cuda
}  // end namespace proxy
}  // end namespace idg
