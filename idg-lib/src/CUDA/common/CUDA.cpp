// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <string>

#include <cuda.h>

#include "CUDA.h"

#include "InstanceCUDA.h"

using namespace idg::kernel::cuda;

namespace idg {
namespace proxy {
namespace cuda {
CUDA::CUDA(ProxyInfo info)
    : power_meter_(pmt::get_power_meter(pmt::sensor_host)), mInfo(info) {
#if defined(DEBUG)
  std::cout << "CUDA::" << __func__ << std::endl;
#endif

  cu::init();
  init_devices();
  print_devices();
  print_compiler_flags();
};

CUDA::~CUDA() {
  // CUDA memory should be free'ed before CUDA devices and
  // contexts are free'ed, hence the explicit calls here.
  free_unified_grid();
  free_buffers_wtiling();
  free_memory();
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
    devices.emplace_back(new InstanceCUDA(mInfo, device_numbers[i]));
  }
}

void CUDA::print_devices() {
  std::cout << "Devices: " << std::endl;
  for (std::unique_ptr<InstanceCUDA>& device : devices) {
    std::cout << *device;
  }
  std::cout << std::endl;
}

void CUDA::print_compiler_flags() {
  std::cout << "Compiler flags: " << std::endl;
  for (std::unique_ptr<InstanceCUDA>& device : devices) {
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
  const cu::Context& context = get_device(0).get_context();
  return std::unique_ptr<auxiliary::Memory>(new cu::HostMemory(context, bytes));
}

int CUDA::initialize_jobs(
    const int nr_baselines, const int nr_timesteps, const int nr_channels,
    const int subgrid_size, const size_t bytes_free, const Plan& plan,
    const aocommon::xt::Span<std::complex<float>, 4>& visibilities,
    const aocommon::xt::Span<UVW<float>, 2>& uvw,
    std::vector<JobData>& jobs) const {
  int jobsize = 0;
  size_t bytes_required = 0;

  do {
    jobsize = jobsize == 0 ? nr_baselines : jobsize * 0.9;
    int max_nr_subgrids = plan.get_max_nr_subgrids(jobsize);
    bytes_required = 0;
    bytes_required +=
        2 * auxiliary::sizeof_visibilities(jobsize, nr_timesteps, nr_channels,
                                           nr_correlations);
    bytes_required += 2 * auxiliary::sizeof_uvw(jobsize, nr_timesteps);
    bytes_required += 2 * auxiliary::sizeof_subgrids(
                              max_nr_subgrids, subgrid_size, nr_correlations);
    bytes_required += 2 * auxiliary::sizeof_metadata(max_nr_subgrids);
  } while (bytes_required > bytes_free);
#if defined(DEBUG)
  std::cout << "jobsize = " << jobsize << std::endl;
#endif

  jobs.clear();

  for (int bl = 0; bl < nr_baselines; bl += jobsize) {
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
    job.uvw_ptr = &uvw(first_bl, 0);
    job.visibilities_ptr =
        const_cast<std::complex<float>*>(&visibilities(first_bl, 0, 0, 0));
    jobs.push_back(job);
  }

  return jobsize;
}

void CUDA::free_buffers_wtiling() {
  m_buffers_wtiling.d_tiles.reset();
  m_buffers_wtiling.d_padded_tiles.reset();
  m_buffers_wtiling.h_tiles.reset();
  m_buffers_wtiling.d_patches.clear();
}

void CUDA::free_unified_grid() { unified_grid_.Reset(); }

}  // end namespace cuda
}  // end namespace proxy
}  // end namespace idg
