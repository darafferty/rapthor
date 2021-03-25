// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "GenericOptimized.h"

#include <cuda.h>
#include <cudaProfiler.h>

#include <algorithm>  // max_element
#include <mutex>
#include <csignal>

#include "InstanceCUDA.h"

using namespace idg::proxy::cuda;
using namespace idg::proxy::cpu;
using namespace idg::kernel::cpu;
using namespace idg::kernel::cuda;
using namespace powersensor;

namespace idg {
namespace proxy {
namespace hybrid {

// Constructor
GenericOptimized::GenericOptimized() : CUDA(default_info()) {
#if defined(DEBUG)
  std::cout << "GenericOptimized::" << __func__ << std::endl;
#endif

  // Initialize cpu proxy
  cpuProxy = new idg::proxy::cpu::Optimized();

  omp_set_nested(true);

  cuProfilerStart();
}

// Destructor
GenericOptimized::~GenericOptimized() {
#if defined(DEBUG)
  std::cout << "GenericOptimized::" << __func__ << std::endl;
#endif

  delete cpuProxy;
  cuProfilerStop();
}

/*
 * FFT
 */
void GenericOptimized::do_transform(DomainAtoDomainB direction) {
#if defined(DEBUG)
  std::cout << "GenericOptimized::" << __func__ << std::endl;
  std::cout << "Transform direction: " << direction << std::endl;
#endif

  cpuProxy->transform(direction);
}  // end transform

/*
 * Gridding
 */
void GenericOptimized::run_gridding(
    const Plan& plan, const Array1D<float>& frequencies,
    const Array3D<Visibility<std::complex<float>>>& visibilities,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines, Grid& grid,
    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterms_offsets,
    const Array2D<float>& spheroidal) {
#if defined(DEBUG)
  std::cout << "GenericOptimized::" << __func__ << std::endl;
#endif

  InstanceCUDA& device = get_device(0);
  const cu::Context& context = device.get_context();

  InstanceCPU& cpuKernels = cpuProxy->get_kernels();

  // Arguments
  auto nr_baselines = visibilities.get_z_dim();
  auto nr_timesteps = visibilities.get_y_dim();
  auto nr_channels = visibilities.get_x_dim();
  auto nr_stations = aterms.get_z_dim();
  auto grid_size = grid.get_x_dim();
  auto cell_size = plan.get_cell_size();
  auto image_size = cell_size * grid_size;
  auto subgrid_size = plan.get_subgrid_size();
  auto w_step = plan.get_w_step();
  auto& shift = plan.get_shift();

  WTileUpdateSet wtile_flush_set = plan.get_wtile_flush_set();

  // Configuration
  const unsigned nr_devices = get_num_devices();
  int device_id = 0;  // only one GPU is used
  int jobsize = m_gridding_state.jobsize[0];

  // Page-locked host memory
  cu::RegisteredMemory h_metadata(context, (void*)plan.get_metadata_ptr(),
                                  plan.get_sizeof_metadata());
  auto max_nr_subgrids = plan.get_max_nr_subgrids(jobsize);
  auto sizeof_subgrids =
      auxiliary::sizeof_subgrids(max_nr_subgrids, subgrid_size);
  cu::HostMemory& h_subgrids = device.allocate_host_subgrids(sizeof_subgrids);

  // Performance measurements
  report.initialize(nr_channels, subgrid_size, grid_size);
  device.set_report(report);
  cpuKernels.set_report(report);
  std::vector<State> startStates(nr_devices + 1);
  std::vector<State> endStates(nr_devices + 1);

  // Events
  std::vector<std::unique_ptr<cu::Event>> inputCopied;
  std::vector<std::unique_ptr<cu::Event>> gpuFinished;
  std::vector<std::unique_ptr<cu::Event>> outputCopied;
  for (unsigned bl = 0; bl < nr_baselines; bl += jobsize) {
    inputCopied.push_back(std::unique_ptr<cu::Event>(new cu::Event(context)));
    gpuFinished.push_back(std::unique_ptr<cu::Event>(new cu::Event(context)));
    outputCopied.push_back(std::unique_ptr<cu::Event>(new cu::Event(context)));
  }

  // Load streams
  cu::Stream& executestream = device.get_execute_stream();
  cu::Stream& htodstream = device.get_htod_stream();
  cu::Stream& dtohstream = device.get_dtoh_stream();

  // Start performance measurement
  startStates[device_id] = device.measure();
  startStates[nr_devices] = hostPowerSensor->read();

  // Locks to make the GPU wait
  std::vector<std::mutex> locks_gpu(jobs.size());
  for (auto& lock : locks_gpu) {
    lock.lock();
  }

  // Locks to make the CPU wait
  std::vector<std::mutex> locks_cpu(jobs.size());
  for (auto& lock : locks_cpu) {
    lock.lock();
  }

  // Start asynchronous computation on the host
  std::thread host_thread = std::thread([&]() {
    int subgrid_offset = 0;
    for (unsigned job_id = 0; job_id < jobs.size(); job_id++) {
      // Get parameters for current job
      auto current_nr_subgrids = jobs[job_id].current_nr_subgrids;
      void* metadata_ptr = jobs[job_id].metadata_ptr;
      std::complex<float>* grid_ptr = grid.data();
      unsigned local_id = job_id % 2;

      // Load memory objects
      cu::DeviceMemory& d_subgrids = *m_buffers.d_subgrids_[local_id];

      // Wait for scaler to finish
      locks_cpu[job_id].lock();

      // Copy subgrid to host
      dtohstream.waitEvent(*gpuFinished[job_id]);
      auto sizeof_subgrids =
          auxiliary::sizeof_subgrids(current_nr_subgrids, subgrid_size);
      dtohstream.memcpyDtoHAsync(h_subgrids, d_subgrids, sizeof_subgrids);
      dtohstream.record(*outputCopied[job_id]);

      // Wait for subgrids to be copied
      outputCopied[job_id]->synchronize();

      // Run adder on host
      cu::Marker marker_adder("run_adder", cu::Marker::blue);
      marker_adder.start();
      if (w_step == 0.0) {
        cpuKernels.run_adder(current_nr_subgrids, grid_size, subgrid_size,
                             metadata_ptr, h_subgrids, grid_ptr);
      } else if (plan.get_use_wtiles()) {
        cpuKernels.run_adder_wtiles(
            current_nr_subgrids, grid_size, subgrid_size, image_size, w_step,
            shift.data(), subgrid_offset, wtile_flush_set, metadata_ptr,
            h_subgrids, grid_ptr);
        subgrid_offset += current_nr_subgrids;
      } else {
        cpuKernels.run_adder_wstack(current_nr_subgrids, grid_size,
                                    subgrid_size, metadata_ptr, h_subgrids,
                                    grid_ptr);
      }
      marker_adder.end();

      // Report performance
      device.enqueue_report(dtohstream, jobs[job_id].current_nr_timesteps,
                            jobs[job_id].current_nr_subgrids);

      // Signal that the subgrids are added
      locks_gpu[job_id].unlock();
    }
  });

  // Iterate all jobs
  for (unsigned job_id = 0; job_id < jobs.size(); job_id++) {
    // Id for double-buffering
    unsigned local_id = job_id % 2;
    unsigned job_id_next = job_id + 1;
    unsigned local_id_next = (local_id + 1) % 2;

    // Get parameters for current job
    auto current_time_offset = jobs[job_id].current_time_offset;
    auto current_nr_baselines = jobs[job_id].current_nr_baselines;
    auto current_nr_subgrids = jobs[job_id].current_nr_subgrids;
    void* metadata_ptr = jobs[job_id].metadata_ptr;
    void* uvw_ptr = jobs[job_id].uvw_ptr;
    void* visibilities_ptr = jobs[job_id].visibilities_ptr;

    // Load memory objects
    cu::DeviceMemory& d_visibilities = *m_buffers.d_visibilities_[local_id];
    cu::DeviceMemory& d_uvw = *m_buffers.d_uvw_[local_id];
    cu::DeviceMemory& d_subgrids = *m_buffers.d_subgrids_[local_id];
    cu::DeviceMemory& d_metadata = *m_buffers.d_metadata_[local_id];
    cu::DeviceMemory& d_wavenumbers = *m_buffers.d_wavenumbers;
    cu::DeviceMemory& d_spheroidal = *m_buffers.d_spheroidal;
    cu::DeviceMemory& d_aterms = *m_buffers.d_aterms;
    cu::DeviceMemory& d_aterms_indices = *m_buffers.d_aterms_indices_[0];
    cu::DeviceMemory& d_avg_aterm = *m_buffers.d_avg_aterm;

    // Copy input data for first job to device
    if (job_id == 0) {
      auto sizeof_visibilities = auxiliary::sizeof_visibilities(
          current_nr_baselines, nr_timesteps, nr_channels);
      auto sizeof_uvw =
          auxiliary::sizeof_uvw(current_nr_baselines, nr_timesteps);
      auto sizeof_metadata = auxiliary::sizeof_metadata(current_nr_subgrids);
      htodstream.memcpyHtoDAsync(d_visibilities, visibilities_ptr,
                                 sizeof_visibilities);
      htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, sizeof_uvw);
      htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, sizeof_metadata);
      htodstream.record(*inputCopied[job_id]);
    }

    // Copy input data for next job
    if (job_id_next < jobs.size()) {
      // Load memory objects
      cu::DeviceMemory& d_visibilities_next = *m_buffers.d_visibilities_[local_id_next];
      cu::DeviceMemory& d_uvw_next = *m_buffers.d_uvw_[local_id_next];
      cu::DeviceMemory& d_metadata_next = *m_buffers.d_metadata_[local_id_next];

      // Get parameters for next job
      auto nr_baselines_next = jobs[job_id_next].current_nr_baselines;
      auto nr_subgrids_next = jobs[job_id_next].current_nr_subgrids;
      void* metadata_ptr_next = jobs[job_id_next].metadata_ptr;
      void* uvw_ptr_next = jobs[job_id_next].uvw_ptr;
      void* visibilities_ptr_next = jobs[job_id_next].visibilities_ptr;

      // Copy input data to device
      auto sizeof_visibilities_next = auxiliary::sizeof_visibilities(
          nr_baselines_next, nr_timesteps, nr_channels);
      auto sizeof_uvw_next =
          auxiliary::sizeof_uvw(nr_baselines_next, nr_timesteps);
      auto sizeof_metadata_next = auxiliary::sizeof_metadata(nr_subgrids_next);
      htodstream.memcpyHtoDAsync(d_visibilities_next, visibilities_ptr_next,
                                 sizeof_visibilities_next);
      htodstream.memcpyHtoDAsync(d_uvw_next, uvw_ptr_next, sizeof_uvw_next);
      htodstream.memcpyHtoDAsync(d_metadata_next, metadata_ptr_next,
                                 sizeof_metadata_next);
      htodstream.record(*inputCopied[job_id_next]);
    }

    // Wait for output buffer to be free
    if (job_id > 1) {
      locks_gpu[job_id - 2].lock();
    }

    // Initialize subgrids to zero
    d_subgrids.zero(executestream);

    // Wait for input to be copied
    executestream.waitEvent(*inputCopied[job_id]);

    // Launch gridder kernel
    device.launch_gridder(current_time_offset, current_nr_subgrids, grid_size,
                          subgrid_size, image_size, w_step, nr_channels,
                          nr_stations, shift(0), shift(1), d_uvw,
                          d_wavenumbers, d_visibilities, d_spheroidal,
                          d_aterms, d_aterms_indices, d_avg_aterm,
                          d_metadata, d_subgrids);

    // Launch FFT
    device.launch_subgrid_fft(d_subgrids, current_nr_subgrids,
                              FourierDomainToImageDomain);

    // Launch scaler
    device.launch_scaler(current_nr_subgrids, subgrid_size, d_subgrids);
    executestream.record(*gpuFinished[job_id]);

    // Wait for scalar to finish
    gpuFinished[job_id]->synchronize();

    // Signal that the subgrids are computed
    locks_cpu[job_id].unlock();
  }  // end for bl

  // Wait for host thread
  if (host_thread.joinable()) {
    host_thread.join();
  }

  // End performance measurement
  endStates[device_id] = device.measure();
  endStates[nr_devices] = hostPowerSensor->read();
  report.update_host(startStates[nr_devices], endStates[nr_devices]);

  // Update report
  auto total_nr_subgrids = plan.get_nr_subgrids();
  auto total_nr_timesteps = plan.get_nr_timesteps();
  auto total_nr_visibilities = plan.get_nr_visibilities();
  report.print_total(total_nr_timesteps, total_nr_subgrids);
  report.print_visibilities(auxiliary::name_gridding, total_nr_visibilities);
}  // end run_gridding

void GenericOptimized::do_gridding(
    const Plan& plan, const Array1D<float>& frequencies,
    const Array3D<Visibility<std::complex<float>>>& visibilities,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterms_offsets,
    const Array2D<float>& spheroidal) {
#if defined(DEBUG)
  std::cout << "GenericOptimized::" << __func__ << std::endl;
#endif

#if defined(DEBUG)
  std::clog << "### Initialize gridding" << std::endl;
#endif
  CUDA::initialize(plan, frequencies, visibilities, uvw, baselines, aterms,
                   aterms_offsets, spheroidal);

#if defined(DEBUG)
  std::clog << "### Run gridding" << std::endl;
#endif
  run_gridding(plan, frequencies, visibilities, uvw, baselines, *m_grid, aterms,
               aterms_offsets, spheroidal);

#if defined(DEBUG)
  std::clog << "### Finish gridding" << std::endl;
#endif
}  // end do_gridding

/*
 * Degridding
 */
void GenericOptimized::run_degridding(
    const Plan& plan, const Array1D<float>& frequencies,
    Array3D<Visibility<std::complex<float>>>& visibilities,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Grid& grid, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterms_offsets,
    const Array2D<float>& spheroidal) {
#if defined(DEBUG)
  std::cout << "GenericOptimized::" << __func__ << std::endl;
#endif

  InstanceCUDA& device = get_device(0);
  const cu::Context& context = device.get_context();

  InstanceCPU& cpuKernels = cpuProxy->get_kernels();

  // Arguments
  auto nr_baselines = visibilities.get_z_dim();
  auto nr_timesteps = visibilities.get_y_dim();
  auto nr_channels = visibilities.get_x_dim();
  auto nr_stations = aterms.get_z_dim();
  auto grid_size = grid.get_x_dim();
  auto cell_size = plan.get_cell_size();
  auto image_size = cell_size * grid_size;
  auto subgrid_size = plan.get_subgrid_size();
  auto w_step = plan.get_w_step();
  auto& shift = plan.get_shift();

  WTileUpdateSet wtile_initialize_set = plan.get_wtile_initialize_set();

  // Configuration
  const unsigned nr_devices = get_num_devices();
  int device_id = 0;  // only one GPU is used
  int jobsize = m_gridding_state.jobsize[0];

  // Page-locked host memory
  cu::RegisteredMemory h_metadata(context, (void*)plan.get_metadata_ptr(),
                                  plan.get_sizeof_metadata());
  auto max_nr_subgrids = plan.get_max_nr_subgrids(jobsize);
  auto sizeof_subgrids =
      auxiliary::sizeof_subgrids(max_nr_subgrids, subgrid_size);
  cu::HostMemory& h_subgrids = device.allocate_host_subgrids(sizeof_subgrids);

  // Performance measurements
  report.initialize(nr_channels, subgrid_size, grid_size);
  device.set_report(report);
  cpuKernels.set_report(report);
  std::vector<State> startStates(nr_devices + 1);
  std::vector<State> endStates(nr_devices + 1);

  // Events
  std::vector<std::unique_ptr<cu::Event>> inputCopied;
  std::vector<std::unique_ptr<cu::Event>> gpuFinished;
  std::vector<std::unique_ptr<cu::Event>> outputCopied;
  for (unsigned bl = 0; bl < nr_baselines; bl += jobsize) {
    inputCopied.push_back(std::unique_ptr<cu::Event>(new cu::Event(context)));
    gpuFinished.push_back(std::unique_ptr<cu::Event>(new cu::Event(context)));
    outputCopied.push_back(std::unique_ptr<cu::Event>(new cu::Event(context)));
  }

  // Load memory objects
  cu::DeviceMemory& d_wavenumbers = *m_buffers.d_wavenumbers;
  cu::DeviceMemory& d_spheroidal = *m_buffers.d_spheroidal;
  cu::DeviceMemory& d_aterms = *m_buffers.d_aterms;
  cu::DeviceMemory& d_aterms_indices = *m_buffers.d_aterms_indices_[0];

  // Load streams
  cu::Stream& executestream = device.get_execute_stream();
  cu::Stream& htodstream = device.get_htod_stream();
  cu::Stream& dtohstream = device.get_dtoh_stream();

  // Start performance measurement
  startStates[device_id] = device.measure();
  startStates[nr_devices] = hostPowerSensor->read();

  // Locks to make the GPU wait
  std::vector<std::mutex> locks_gpu(jobs.size());
  for (auto& lock : locks_gpu) {
    lock.lock();
  }

  // Locks to make the CPU wait
  std::vector<std::mutex> locks_cpu(jobs.size());
  for (auto& lock : locks_cpu) {
    lock.lock();
  }

  // Start host thread to create subgrids
  std::thread host_thread = std::thread([&] {
    int subgrid_offset = 0;
    for (unsigned job_id = 0; job_id < jobs.size(); job_id++) {
      // Get parameters for current job
      auto current_nr_subgrids = jobs[job_id].current_nr_subgrids;
      void* metadata_ptr = jobs[job_id].metadata_ptr;
      std::complex<float>* grid_ptr = grid.data();
      unsigned local_id = job_id % 2;

      // Load memory objects
      cu::DeviceMemory& d_subgrids = *m_buffers.d_subgrids_[local_id];

      // Wait for input buffer to be free
      if (job_id > 0) {
        locks_cpu[job_id - 1].lock();
      }

      // Run splitter kernel
      cu::Marker marker_splitter("run_splitter", cu::Marker::blue);
      marker_splitter.start();

      if (w_step == 0.0) {
        cpuKernels.run_splitter(current_nr_subgrids, grid_size, subgrid_size,
                                metadata_ptr, h_subgrids, grid_ptr);
      } else if (plan.get_use_wtiles()) {
        cpuKernels.run_splitter_wtiles(
            current_nr_subgrids, grid_size, subgrid_size, image_size, w_step,
            shift.data(), subgrid_offset, wtile_initialize_set, metadata_ptr,
            h_subgrids, grid_ptr);
        subgrid_offset += current_nr_subgrids;
      } else {
        cpuKernels.run_splitter_wstack(current_nr_subgrids, grid_size,
                                       subgrid_size, metadata_ptr, h_subgrids,
                                       grid_ptr);
      }

      marker_splitter.end();

      // Copy subgrids to device
      auto sizeof_subgrids =
          auxiliary::sizeof_subgrids(current_nr_subgrids, subgrid_size);
      htodstream.memcpyHtoDAsync(d_subgrids, h_subgrids, sizeof_subgrids);

      // Wait for subgrids to be copied
      htodstream.synchronize();

      // Unlock this job
      locks_gpu[job_id].unlock();
    }
  });  // end host thread

  // Iterate all jobs
  for (unsigned job_id = 0; job_id < jobs.size(); job_id++) {
    // Id for double-buffering
    unsigned local_id = job_id % 2;
    unsigned job_id_next = job_id + 1;
    unsigned local_id_next = (local_id + 1) % 2;

    // Get parameters for current job
    auto current_time_offset = jobs[job_id].current_time_offset;
    auto current_nr_baselines = jobs[job_id].current_nr_baselines;
    auto current_nr_subgrids = jobs[job_id].current_nr_subgrids;
    void* metadata_ptr = jobs[job_id].metadata_ptr;
    void* uvw_ptr = jobs[job_id].uvw_ptr;
    void* visibilities_ptr = jobs[job_id].visibilities_ptr;

    // Load memory objects
    cu::DeviceMemory& d_visibilities = *m_buffers.d_visibilities_[local_id];
    cu::DeviceMemory& d_uvw = *m_buffers.d_uvw_[local_id];
    cu::DeviceMemory& d_subgrids = *m_buffers.d_subgrids_[local_id];
    cu::DeviceMemory& d_metadata = *m_buffers.d_metadata_[local_id];

    // Wait for subgrids to be computed
    locks_gpu[job_id].lock();

    // Copy input data for first job to device
    if (job_id == 0) {
      auto sizeof_uvw =
          auxiliary::sizeof_uvw(current_nr_baselines, nr_timesteps);
      auto sizeof_metadata = auxiliary::sizeof_metadata(current_nr_subgrids);
      htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, sizeof_uvw);
      htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, sizeof_metadata);
      htodstream.record(*inputCopied[job_id]);
    }

    // Copy input data for next job
    if (job_id_next < jobs.size()) {
      // Load memory objects
      cu::DeviceMemory& d_uvw_next = *m_buffers.d_uvw_[local_id_next];
      cu::DeviceMemory& d_metadata_next = *m_buffers.d_metadata_[local_id_next];

      // Get parameters for next job
      auto nr_baselines_next = jobs[job_id_next].current_nr_baselines;
      auto nr_subgrids_next = jobs[job_id_next].current_nr_subgrids;
      void* metadata_ptr_next = jobs[job_id_next].metadata_ptr;
      void* uvw_ptr_next = jobs[job_id_next].uvw_ptr;

      // Copy input data to device
      auto sizeof_uvw_next =
          auxiliary::sizeof_uvw(nr_baselines_next, nr_timesteps);
      auto sizeof_metadata_next = auxiliary::sizeof_metadata(nr_subgrids_next);
      htodstream.memcpyHtoDAsync(d_uvw_next, uvw_ptr_next, sizeof_uvw_next);
      htodstream.memcpyHtoDAsync(d_metadata_next, metadata_ptr_next,
                                 sizeof_metadata_next);
      htodstream.record(*inputCopied[job_id_next]);
    }

    // Wait for input to be copied
    executestream.waitEvent(*inputCopied[job_id]);

    // Wait for output buffer to be free
    if (job_id > 1) {
      executestream.waitEvent(*outputCopied[job_id - 2]);
    }

    // Initialize visibilities to zero
    d_visibilities.zero(executestream);

    // Launch FFT
    device.launch_subgrid_fft(d_subgrids, current_nr_subgrids,
                              ImageDomainToFourierDomain);

    // Launch degridder kernel
    device.launch_degridder(current_time_offset, current_nr_subgrids, grid_size,
                        subgrid_size, image_size, w_step, nr_channels,
                        nr_stations, shift(0), shift(1), d_uvw,
                        d_wavenumbers, d_visibilities, d_spheroidal,
                        d_aterms, d_aterms_indices,
                        d_metadata, d_subgrids);
    executestream.record(*gpuFinished[job_id]);

    // Signal that the input buffer is free
    inputCopied[job_id]->synchronize();
    locks_cpu[job_id].unlock();

    // Wait for degridder to finish
    gpuFinished[job_id]->synchronize();

    // Copy visibilities to host
    dtohstream.waitEvent(*gpuFinished[job_id]);
    auto sizeof_visibilities = auxiliary::sizeof_visibilities(
        current_nr_baselines, nr_timesteps, nr_channels);
    dtohstream.memcpyDtoHAsync(visibilities_ptr, d_visibilities,
                               sizeof_visibilities);
    dtohstream.record(*outputCopied[job_id]);

    // Report performance
    device.enqueue_report(executestream, jobs[job_id].current_nr_timesteps,
                          jobs[job_id].current_nr_subgrids);
  }  // end for bl

  // Wait for host thread
  if (host_thread.joinable()) {
    host_thread.join();
  }

  // Wait for all visibilities to be copied
  dtohstream.synchronize();

  // End performance measurement
  endStates[device_id] = device.measure();
  endStates[nr_devices] = hostPowerSensor->read();
  report.update_host(startStates[nr_devices], endStates[nr_devices]);

  // Update report
  auto total_nr_subgrids = plan.get_nr_subgrids();
  auto total_nr_timesteps = plan.get_nr_timesteps();
  auto total_nr_visibilities = plan.get_nr_visibilities();
  report.print_total(total_nr_timesteps, total_nr_subgrids);
  report.print_visibilities(auxiliary::name_degridding, total_nr_visibilities);
}  // end run_degridding

void GenericOptimized::do_degridding(
    const Plan& plan, const Array1D<float>& frequencies,
    Array3D<Visibility<std::complex<float>>>& visibilities,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterms_offsets,
    const Array2D<float>& spheroidal) {
#if defined(DEBUG)
  std::cout << "GenericOptimized::" << __func__ << std::endl;
  std::clog << "### Initialize degridding" << std::endl;
#endif
  CUDA::initialize(plan, frequencies, visibilities, uvw, baselines, aterms,
                   aterms_offsets, spheroidal);

#if defined(DEBUG)
  std::clog << "### Run degridding" << std::endl;
#endif
  run_degridding(plan, frequencies, visibilities, uvw, baselines, *m_grid,
                 aterms, aterms_offsets, spheroidal);

#if defined(DEBUG)
  std::clog << "### Finish degridding" << std::endl;
#endif
}  // end do_degridding

void GenericOptimized::do_calibrate_init(
    std::vector<std::unique_ptr<Plan>>&& plans,
    const Array1D<float>& frequencies,
    Array4D<Visibility<std::complex<float>>>&& visibilities,
    Array4D<Visibility<float>>&& weights, Array3D<UVW<float>>&& uvw,
    Array2D<std::pair<unsigned int, unsigned int>>&& baselines,
    const Array2D<float>& spheroidal) {
  InstanceCPU& cpuKernels = cpuProxy->get_kernels();
  cpuKernels.set_report(report);

  Array1D<float> wavenumbers = compute_wavenumbers(frequencies);

  // Arguments
  auto nr_antennas = plans.size();
  auto grid_size = m_grid->get_x_dim();
  auto image_size = m_cache_state.cell_size * grid_size;
  auto w_step = m_cache_state.w_step;
  auto subgrid_size = m_cache_state.subgrid_size;
  auto nr_baselines = visibilities.get_z_dim();
  auto nr_timesteps = visibilities.get_y_dim();
  auto nr_channels = visibilities.get_x_dim();
  auto max_nr_terms = m_calibrate_max_nr_terms;
  auto nr_correlations = 4;
  auto& shift = m_cache_state.shift;

  // Allocate subgrids for all antennas
  std::vector<Array4D<std::complex<float>>> subgrids;
  subgrids.reserve(nr_antennas);

  // Start performance measurement
  report.initialize();
  powersensor::State states[2];
  states[0] = hostPowerSensor->read();

  // Load device
  InstanceCUDA& device = get_device(0);
  device.set_report(report);

  // Load stream
  cu::Stream& htodstream = device.get_htod_stream();

  // Find max number of subgrids
  unsigned int max_nr_subgrids = 0;

  // Initialize buffers
  m_buffers.d_metadata_.resize(0);
  m_buffers.d_subgrids_.resize(0);
  m_buffers.d_visibilities_.resize(0);
  m_buffers.d_weights_.resize(0);
  m_buffers.d_uvw_.resize(0);
  m_buffers.d_aterms_indices_.resize(0);

  // Create subgrids for every antenna
  for (unsigned int antenna_nr = 0; antenna_nr < nr_antennas; antenna_nr++) {
    // Allocate subgrids for current antenna
    unsigned int nr_subgrids = plans[antenna_nr]->get_nr_subgrids();
    Array4D<std::complex<float>> subgrids_(nr_subgrids, nr_polarizations,
                                           subgrid_size, subgrid_size);

    if (nr_subgrids > max_nr_subgrids) {
      max_nr_subgrids = nr_subgrids;
    }

    // Get data pointers
    void* metadata_ptr = (void*)plans[antenna_nr]->get_metadata_ptr();
    void* subgrids_ptr = subgrids_.data();
    std::complex<float>* grid_ptr = m_grid->data();
    void* aterm_idx_ptr = (void*)plans[antenna_nr]->get_aterm_indices_ptr();

    // Splitter kernel
    if (w_step == 0.0) {
      cpuKernels.run_splitter(nr_subgrids, grid_size, subgrid_size,
                              metadata_ptr, subgrids_ptr, grid_ptr);
    } else if (plans[antenna_nr]->get_use_wtiles()) {
      WTileUpdateSet wtile_initialize_set =
          plans[antenna_nr]->get_wtile_initialize_set();
      cpuKernels.run_splitter_wtiles(
          nr_subgrids, grid_size, subgrid_size, image_size, w_step,
          shift.data(), 0 /* subgrid_offset */, wtile_initialize_set,
          metadata_ptr, subgrids_ptr, grid_ptr);
    } else {
      cpuKernels.run_splitter_wstack(nr_subgrids, grid_size, subgrid_size,
                                     metadata_ptr, subgrids_ptr, grid_ptr);
    }

    // FFT kernel
    cpuKernels.run_subgrid_fft(grid_size, subgrid_size, nr_subgrids,
                               subgrids_ptr, CUFFT_FORWARD);

    // Apply spheroidal
    for (int i = 0; i < (int)nr_subgrids; i++) {
      for (int pol = 0; pol < nr_polarizations; pol++) {
        for (int j = 0; j < subgrid_size; j++) {
          for (int k = 0; k < subgrid_size; k++) {
            int y = (j + (subgrid_size / 2)) % subgrid_size;
            int x = (k + (subgrid_size / 2)) % subgrid_size;
            subgrids_(i, pol, y, x) *= spheroidal(j, k);
          }
        }
      }
    }

    // Allocate and initialize device memory for current antenna
    void* visibilities_ptr = visibilities.data(antenna_nr);
    void* weights_ptr = weights.data(antenna_nr);
    void* uvw_ptr = uvw.data(antenna_nr);
    auto sizeof_metadata = auxiliary::sizeof_metadata(nr_subgrids);
    auto sizeof_subgrids =
        auxiliary::sizeof_subgrids(nr_subgrids, subgrid_size);
    auto sizeof_visibilities =
        auxiliary::sizeof_visibilities(nr_baselines, nr_timesteps, nr_channels);
    auto sizeof_weights =
        auxiliary::sizeof_weights(nr_baselines, nr_timesteps, nr_channels);
    auto sizeof_uvw = auxiliary::sizeof_uvw(nr_baselines, nr_timesteps);
    auto sizeof_aterm_idx =
        auxiliary::sizeof_aterms_indices(nr_baselines, nr_timesteps);
    m_buffers.d_metadata_[antenna_nr]->resize(sizeof_metadata);
    m_buffers.d_subgrids_[antenna_nr]->resize(sizeof_subgrids);
    m_buffers.d_visibilities_[antenna_nr]->resize(sizeof_visibilities);
    m_buffers.d_weights_[antenna_nr]->resize(sizeof_weights);
    m_buffers.d_uvw_[antenna_nr]->resize(sizeof_uvw);
    m_buffers.d_aterms_indices_[antenna_nr]->resize(sizeof_aterm_idx);
    cu::DeviceMemory& d_metadata = *m_buffers.d_metadata_[antenna_nr];
    cu::DeviceMemory& d_subgrids = *m_buffers.d_subgrids_[antenna_nr];
    cu::DeviceMemory& d_visibilities = *m_buffers.d_visibilities_[antenna_nr];
    cu::DeviceMemory& d_weights = *m_buffers.d_weights_[antenna_nr];
    cu::DeviceMemory& d_uvw = *m_buffers.d_uvw_[antenna_nr];
    cu::DeviceMemory& d_aterm_idx = *m_buffers.d_aterms_indices_[antenna_nr];
    htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, sizeof_metadata);
    htodstream.memcpyHtoDAsync(d_subgrids, subgrids_ptr, sizeof_subgrids);
    htodstream.memcpyHtoDAsync(d_visibilities, visibilities_ptr, sizeof_visibilities);
    htodstream.memcpyHtoDAsync(d_weights, weights_ptr, sizeof_weights);
    htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, sizeof_uvw);
    htodstream.memcpyHtoDAsync(d_aterm_idx, aterm_idx_ptr, sizeof_aterm_idx);
    htodstream.synchronize();
  }  // end for antennas

  // End performance measurement
  states[1] = hostPowerSensor->read();
  report.update_host(states[0], states[1]);
  report.print_total(0, 0);

  // Set calibration state member variables
  m_calibrate_state.plans = std::move(plans);
  m_calibrate_state.nr_baselines = nr_baselines;
  m_calibrate_state.nr_timesteps = nr_timesteps;
  m_calibrate_state.nr_channels = nr_channels;

  // Initialize wavenumbers
  m_buffers.d_wavenumbers->resize(wavenumbers.bytes());
  cu::DeviceMemory& d_wavenumbers = *m_buffers.d_wavenumbers;
  htodstream.memcpyHtoDAsync(d_wavenumbers, wavenumbers.data(),
                             wavenumbers.bytes());

  // Allocate device memory for l,m,n and phase offset
  auto sizeof_lmnp =
      max_nr_subgrids * subgrid_size * subgrid_size * 4 * sizeof(float);
  m_buffers.d_lmnp->resize(sizeof_lmnp);

  // Allocate memory for sums (horizontal and vertical)
  auto total_nr_timesteps = nr_baselines * nr_timesteps;
  auto sizeof_sums = max_nr_terms * nr_correlations * total_nr_timesteps *
                     nr_channels * sizeof(std::complex<float>);
  for (unsigned int i = 0; i < 2; i++) {
    m_buffers.d_sums_[i]->resize(sizeof_sums);
  }
}

void GenericOptimized::do_calibrate_update(
    const int antenna_nr, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array4D<Matrix2x2<std::complex<float>>>& aterm_derivatives,
    Array3D<double>& hessian, Array2D<double>& gradient, double& residual) {
  // Arguments
  auto nr_subgrids = m_calibrate_state.plans[antenna_nr]->get_nr_subgrids();
  auto nr_baselines = m_calibrate_state.nr_baselines;
  auto nr_timesteps = m_calibrate_state.nr_timesteps;
  auto nr_channels = m_calibrate_state.nr_channels;
  auto nr_terms = aterm_derivatives.get_z_dim();
  auto subgrid_size = aterms.get_y_dim();
  auto nr_timeslots = aterms.get_w_dim();
  auto nr_stations = aterms.get_z_dim();
  auto grid_size = m_grid->get_y_dim();
  auto image_size = m_cache_state.cell_size * grid_size;
  auto w_step = m_cache_state.w_step;
  auto nr_correlations = 4;

  // Performance measurement
  if (antenna_nr == 0) {
    report.initialize(nr_channels, subgrid_size, 0, nr_terms);
  }

  // Start marker
  cu::Marker marker("do_calibrate_update");
  marker.start();

  // Load device
  InstanceCUDA& device = get_device(0);
  const cu::Context& context = device.get_context();

  // Transpose aterms and aterm derivatives
  const unsigned int nr_aterms = nr_stations * nr_timeslots;
  const unsigned int nr_aterm_derivatives = nr_terms * nr_timeslots;
  Array4D<std::complex<float>> aterms_transposed(nr_aterms, nr_correlations,
                                                 subgrid_size, subgrid_size);
  Array4D<std::complex<float>> aterm_derivatives_transposed(
      nr_aterm_derivatives, nr_correlations, subgrid_size, subgrid_size);
  device.transpose_aterm(aterms, aterms_transposed);
  device.transpose_aterm(aterm_derivatives, aterm_derivatives_transposed);

  // Load streams
  cu::Stream& executestream = device.get_execute_stream();
  cu::Stream& htodstream = device.get_htod_stream();
  cu::Stream& dtohstream = device.get_dtoh_stream();

  // Load memory objects
  cu::DeviceMemory& d_wavenumbers = *m_buffers.d_wavenumbers;
  m_buffers.d_aterms->resize(aterms.bytes());
  cu::DeviceMemory& d_aterms = *m_buffers.d_aterms;
  cu::DeviceMemory& d_metadata = *m_buffers.d_metadata_[antenna_nr];
  cu::DeviceMemory& d_subgrids = *m_buffers.d_subgrids_[antenna_nr];
  cu::DeviceMemory& d_visibilities = *m_buffers.d_visibilities_[antenna_nr];
  cu::DeviceMemory& d_weights = *m_buffers.d_weights_[antenna_nr];
  cu::DeviceMemory& d_uvw = *m_buffers.d_uvw_[antenna_nr];
  cu::DeviceMemory& d_sums1 = *m_buffers.d_sums_[0];
  cu::DeviceMemory& d_sums2 = *m_buffers.d_sums_[1];
  cu::DeviceMemory& d_lmnp = *m_buffers.d_lmnp;
  cu::DeviceMemory& d_aterms_idx = *m_buffers.d_aterms_indices_[antenna_nr];

  // Allocate additional data structures
  cu::DeviceMemory d_aterms_deriv(context, aterm_derivatives.bytes());
  cu::DeviceMemory d_hessian(context, hessian.bytes());
  cu::DeviceMemory d_gradient(context, gradient.bytes());
  cu::DeviceMemory d_residual(context, sizeof(double));
  cu::HostMemory h_hessian(context, hessian.bytes());
  cu::HostMemory h_gradient(context, gradient.bytes());
  cu::HostMemory h_residual(context, sizeof(double));

  // Events
  cu::Event inputCopied(context), executeFinished(context),
      outputCopied(context);

  // Copy input data to device
  htodstream.memcpyHtoDAsync(d_aterms, aterms_transposed.data(),
                             aterms_transposed.bytes());
  htodstream.memcpyHtoDAsync(d_aterms_deriv,
                             aterm_derivatives_transposed.data(),
                             aterm_derivatives_transposed.bytes());
  htodstream.memcpyHtoDAsync(d_hessian, hessian.data(), hessian.bytes());
  htodstream.memcpyHtoDAsync(d_gradient, gradient.data(), gradient.bytes());
  htodstream.memcpyHtoDAsync(d_residual, &residual, sizeof(double));
  htodstream.record(inputCopied);

  // Run calibration update step
  executestream.waitEvent(inputCopied);
  auto total_nr_timesteps = nr_baselines * nr_timesteps;
  device.launch_calibrate(nr_subgrids, grid_size, subgrid_size, image_size,
                          w_step, total_nr_timesteps, nr_channels, nr_stations,
                          nr_terms, d_uvw, d_wavenumbers, d_visibilities,
                          d_weights, d_aterms, d_aterms_deriv, d_aterms_idx,
                          d_metadata, d_subgrids, d_sums1, d_sums2, d_lmnp,
                          d_hessian, d_gradient, d_residual);
  executestream.record(executeFinished);

  // Copy output to host
  dtohstream.waitEvent(executeFinished);
  dtohstream.memcpyDtoHAsync(h_hessian, d_hessian, d_hessian.size());
  dtohstream.memcpyDtoHAsync(h_gradient, d_gradient, d_gradient.size());
  dtohstream.memcpyDtoHAsync(h_residual, d_residual, d_residual.size());
  dtohstream.record(outputCopied);

  // Wait for output to finish
  outputCopied.synchronize();

  // Copy output on host
  memcpy(hessian.data(), h_hessian, hessian.bytes());
  memcpy(gradient.data(), h_gradient, gradient.bytes());
  memcpy(&residual, h_residual, sizeof(double));

  // End marker
  marker.end();

  // Performance reporting
  auto nr_visibilities = nr_timesteps * nr_channels;
  report.update_total(nr_subgrids, nr_timesteps, nr_visibilities);
}

void GenericOptimized::do_calibrate_finish() {
  // Performance reporting
  auto nr_antennas = m_calibrate_state.plans.size();
  auto total_nr_timesteps = 0;
  auto total_nr_subgrids = 0;
  for (unsigned int antenna_nr = 0; antenna_nr < nr_antennas; antenna_nr++) {
    total_nr_timesteps +=
        m_calibrate_state.plans[antenna_nr]->get_nr_timesteps();
    total_nr_subgrids += m_calibrate_state.plans[antenna_nr]->get_nr_subgrids();
  }
  report.print_total(total_nr_timesteps, total_nr_subgrids);
  report.print_visibilities(auxiliary::name_calibrate);
}

void GenericOptimized::do_calibrate_init_hessian_vector_product() {
  m_calibrate_state.hessian_vector_product_visibilities =
      Array3D<Visibility<std::complex<float>>>(m_calibrate_state.nr_baselines,
                                               m_calibrate_state.nr_timesteps,
                                               m_calibrate_state.nr_channels);
  std::memset(m_calibrate_state.hessian_vector_product_visibilities.data(), 0,
              m_calibrate_state.hessian_vector_product_visibilities.bytes());
}

void GenericOptimized::do_calibrate_update_hessian_vector_product1(
    const int antenna_nr, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array4D<Matrix2x2<std::complex<float>>>& derivative_aterms,
    const Array2D<float>& parameter_vector) {
  //                 // Arguments
  //                 auto nr_subgrids   =
  //                 m_calibrate_state.plans[antenna_nr]->get_nr_subgrids();
  //                 auto nr_channels   =
  //                 m_calibrate_state.wavenumbers.get_x_dim(); auto nr_terms =
  //                 aterm_derivatives.get_z_dim(); auto subgrid_size  =
  //                 aterms.get_y_dim(); auto nr_stations   =
  //                 aterms.get_z_dim(); auto nr_timeslots  =
  //                 aterms.get_w_dim();
  //
  //                 // Performance measurement
  //                 if (antenna_nr == 0) {
  //                     report.initialize(nr_channels, subgrid_size, 0,
  //                     nr_terms);
  //                 }
  //
  //                 // Data pointers
  //                 auto shift_ptr                     =
  //                 m_calibrate_state.shift.data(); auto wavenumbers_ptr =
  //                 m_calibrate_state.wavenumbers.data(); idg::float2
  //                 *aterm_ptr             = (idg::float2*) aterms.data();
  //                 idg::float2 * aterm_derivative_ptr = (idg::float2*)
  //                 aterm_derivatives.data(); auto aterm_idx_ptr =
  //                 m_calibrate_state.plans[antenna_nr]->get_aterm_indices_ptr();
  //                 auto metadata_ptr                  =
  //                 m_calibrate_state.plans[antenna_nr]->get_metadata_ptr();
  //                 auto uvw_ptr                       =
  //                 m_calibrate_state.uvw.data(antenna_nr); idg::float2
  //                 *visibilities_ptr      = (idg::float2*)
  //                 m_calibrate_state.visibilities.data(antenna_nr); float
  //                 *weights_ptr                 = (float*)
  //                 m_calibrate_state.weights.data(antenna_nr); idg::float2
  //                 *subgrids_ptr          = (idg::float2*)
  //                 m_calibrate_state.subgrids[antenna_nr].data(); idg::float2
  //                 *phasors_ptr           = (idg::float2*)
  //                 m_calibrate_state.phasors[antenna_nr].data(); float
  //                 *parameter_vector_ptr        = (idg::float2*)
  //                 parameter_vector.data();
  //
  //                 int max_nr_timesteps       =
  //                 m_calibrate_state.max_nr_timesteps[antenna_nr];
  //

  // TODO for now call the cpu instance
  InstanceCPU& cpuKernels = cpuProxy->get_kernels();
  cpuKernels.run_calibrate_hessian_vector_product1(
      antenna_nr, aterms, derivative_aterms, parameter_vector);
}

void GenericOptimized::do_calibrate_update_hessian_vector_product2(
    const int station_nr, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array4D<Matrix2x2<std::complex<float>>>& derivative_aterms,
    Array2D<float>& parameter_vector) {
  // TODO for now call the cpu instance
  InstanceCPU& cpuKernels = cpuProxy->get_kernels();
  cpuKernels.run_calibrate_hessian_vector_product2(
      station_nr, aterms, derivative_aterms, parameter_vector);
}

std::unique_ptr<Plan> GenericOptimized::make_plan(
    const int kernel_size, const Array1D<float>& frequencies,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Array1D<unsigned int>& aterms_offsets, Plan::Options options) {
  // Defer call to cpuProxy
  // cpuProxy manages the wtiles state
  // plan will be made accordingly
  return cpuProxy->make_plan(kernel_size, frequencies, uvw, baselines,
                             aterms_offsets, options);
}

void GenericOptimized::set_grid(std::shared_ptr<Grid> grid) {
  // Defer call to cpuProxy
  cpuProxy->set_grid(grid);
  CUDA::set_grid(grid);
}

std::shared_ptr<Grid> GenericOptimized::get_final_grid() {
  // Defer call to cpuProxy
  return cpuProxy->get_final_grid();
}

void GenericOptimized::init_cache(int subgrid_size, float cell_size,
                                  float w_step, const Array1D<float>& shift) {
  // Defer call to cpuProxy
  // cpuProxy manages the wtiles state
  cpuProxy->init_cache(subgrid_size, cell_size, w_step, shift);
  cuda::CUDA::init_cache(subgrid_size, cell_size, w_step, shift);
}

}  // namespace hybrid
}  // namespace proxy
}  // namespace idg
