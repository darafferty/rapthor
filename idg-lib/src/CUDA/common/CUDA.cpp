// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <string>

#include <cudawrappers/cu.hpp>

#include "CUDA.h"
#include "InstanceCUDA.h"

using namespace idg::kernel::cuda;

namespace idg {
namespace proxy {
namespace cuda {
CUDA::CUDA() : power_meter_(pmt::get_power_meter(pmt::sensor_host)) {
#if defined(DEBUG)
  std::cout << "CUDA::" << __func__ << std::endl;
#endif

  cu::init();
  init_device();
  print_device();
};

CUDA::~CUDA() {
  // CUDA memory should be free'ed before CUDA devices and
  // contexts are free'ed, hence the explicit calls here.
  free_buffers_wtiling();
  free_memory();
  free_host_memory();
}

void CUDA::init_device() {
  char* char_cuda_device = getenv("CUDA_DEVICE");
  int cuda_device = char_cuda_device ? atoi(char_cuda_device) : 0;
  device_ = std::make_unique<InstanceCUDA>(cuda_device);
}

void CUDA::print_device() {
  std::cout << "Device: " << std::endl << *device_ << std::endl;
}

InstanceCUDA& CUDA::get_device() const {
  set_context();
  return *device_;
}

void CUDA::set_context() const { device_->get_context().setCurrent(); }

std::unique_ptr<auxiliary::Memory> CUDA::allocate_memory(size_t bytes) {
  set_context();
  h_memory_.emplace_back(new cu::HostMemory(bytes));
  return std::make_unique<auxiliary::Memory>(*h_memory_.back(), bytes);
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

void CUDA::free_host_memory() { h_memory_.clear(); }

}  // end namespace cuda
}  // end namespace proxy
}  // end namespace idg
