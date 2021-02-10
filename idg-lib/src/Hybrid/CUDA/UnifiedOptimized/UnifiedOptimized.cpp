// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "UnifiedOptimized.h"

#include <thread>
#include <mutex>
#include <csignal>

#include <cuda.h>
#include <cudaProfiler.h>

#include "InstanceCUDA.h"
#include "common/Math.h"
#include "common/Index.h"

using namespace idg::proxy::cuda;
using namespace idg::proxy::cpu;
using namespace idg::kernel::cpu;
using namespace idg::kernel::cuda;
using namespace powersensor;

namespace idg {
namespace proxy {
namespace hybrid {

// Constructor
UnifiedOptimized::UnifiedOptimized() :
  CUDA(default_info()),
  cpuProxy(new idg::proxy::cpu::Optimized())
{
#if defined(DEBUG)
  std::cout << "UnifiedOptimized::" << __func__ << std::endl;
#endif

  omp_set_nested(true);

  // Increase the fraction of reserved memory
  set_fraction_reserved(0.4);

  // Enable unified memory
  enable_unified_memory();

  cuProfilerStart();
}

// Destructor
UnifiedOptimized::~UnifiedOptimized() {
#if defined(DEBUG)
  std::cout << "UnifiedOptimized::" << __func__ << std::endl;
#endif

  // Explicitely free the grid before
  // the CUDA Context is destroyed
  if (m_use_unified_memory)
  {
    m_grid.reset();
  }

  cuProfilerStop();
}

/*
 * FFT
 */
void UnifiedOptimized::do_transform(DomainAtoDomainB direction) {
#if defined(DEBUG)
  std::cout << "UnifiedOptimized::" << __func__ << std::endl;
  std::cout << "Transform direction: " << direction << std::endl;
#endif

  cpuProxy->transform(direction);
}  // end transform

/*
 * Gridding
 */
void UnifiedOptimized::run_gridding(
    const Plan& plan, const Array1D<float>& frequencies,
    const Array3D<Visibility<std::complex<float>>>& visibilities,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines, Grid& grid,
    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterms_offsets,
    const Array2D<float>& spheroidal) {
#if defined(DEBUG)
  std::cout << "UnifiedOptimized::" << __func__ << std::endl;
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
    for (unsigned job_id = 0; job_id < jobs.size(); job_id++) {
      // Get parameters for current job
      auto current_nr_subgrids = jobs[job_id].current_nr_subgrids;
      unsigned local_id = job_id % 2;


      // Wait for scaler to finish
      locks_cpu[job_id].lock();

      // Run adder on host
      cu::Marker marker_adder("run_adder", cu::Marker::blue);
      marker_adder.start();

      // Run adder wtiles
      run_subgrids_to_wtiles(local_id, current_nr_subgrids, subgrid_size, image_size, w_step, shift, wtile_flush_set);

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
    cu::DeviceMemory& d_visibilities =
        device.retrieve_device_visibilities(local_id);
    cu::DeviceMemory& d_uvw = device.retrieve_device_uvw(local_id);
    cu::DeviceMemory& d_subgrids = device.retrieve_device_subgrids(local_id);
    cu::DeviceMemory& d_metadata = device.retrieve_device_metadata(local_id);

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
      cu::DeviceMemory& d_visibilities_next =
          device.retrieve_device_visibilities(local_id_next);
      cu::DeviceMemory& d_uvw_next = device.retrieve_device_uvw(local_id_next);
      cu::DeviceMemory& d_metadata_next =
          device.retrieve_device_metadata(local_id_next);

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
                          nr_stations, d_uvw, d_visibilities, d_metadata,
                          d_subgrids);

    // Launch FFT
    device.launch_subgrid_fft(d_subgrids, current_nr_subgrids,
                              FourierDomainToImageDomain);
    executestream.record(*gpuFinished[job_id]);

    // Wait for FFT to finish
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

void UnifiedOptimized::do_gridding(
    const Plan &plan, const Array1D<float> &frequencies,
    const Array3D<Visibility<std::complex<float>>> &visibilities,
    const Array2D<UVW<float>> &uvw,
    const Array1D<std::pair<unsigned int, unsigned int>> &baselines,
    const Array4D<Matrix2x2<std::complex<float>>> &aterms,
    const Array1D<unsigned int> &aterms_offsets,
    const Array2D<float> &spheroidal) {
#if defined(DEBUG)
  std::cout << "UnifiedOptimized::" << __func__ << std::endl;
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
void UnifiedOptimized::run_degridding(
    const Plan& plan, const Array1D<float>& frequencies,
    Array3D<Visibility<std::complex<float>>>& visibilities,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Grid& grid, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterms_offsets,
    const Array2D<float>& spheroidal) {
#if defined(DEBUG)
  std::cout << "UnifiedOptimized::" << __func__ << std::endl;
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
      cu::DeviceMemory& d_subgrids = device.retrieve_device_subgrids(local_id);

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
    cu::DeviceMemory& d_visibilities =
        device.retrieve_device_visibilities(local_id);
    cu::DeviceMemory& d_uvw = device.retrieve_device_uvw(local_id);
    cu::DeviceMemory& d_subgrids = device.retrieve_device_subgrids(local_id);
    cu::DeviceMemory& d_metadata = device.retrieve_device_metadata(local_id);

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
      cu::DeviceMemory& d_uvw_next = device.retrieve_device_uvw(local_id_next);
      cu::DeviceMemory& d_metadata_next =
          device.retrieve_device_metadata(local_id_next);

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
                            nr_stations, d_uvw, d_visibilities, d_metadata,
                            d_subgrids);
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

void UnifiedOptimized::do_degridding(
    const Plan &plan, const Array1D<float> &frequencies,
    Array3D<Visibility<std::complex<float>>> &visibilities,
    const Array2D<UVW<float>> &uvw,
    const Array1D<std::pair<unsigned int, unsigned int>> &baselines,
    const Array4D<Matrix2x2<std::complex<float>>> &aterms,
    const Array1D<unsigned int> &aterms_offsets,
    const Array2D<float> &spheroidal) {
#if defined(DEBUG)
  std::cout << "UnifiedOptimized::" << __func__ << std::endl;
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

std::unique_ptr<Plan> UnifiedOptimized::make_plan(
    const int kernel_size, const Array1D<float>& frequencies,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Array1D<unsigned int>& aterms_offsets, Plan::Options options) {
  if (supports_wtiling() && m_cache_state.w_step != 0.0 &&
      m_wtiles.get_wtile_buffer_size()) {
    options.nr_w_layers = INT_MAX;
    return std::unique_ptr<Plan>(
        new Plan(kernel_size, m_cache_state.subgrid_size, m_grid->get_y_dim(),
                 m_cache_state.cell_size, m_cache_state.shift, frequencies, uvw,
                 baselines, aterms_offsets, m_wtiles, options));
  } else {
    return Proxy::make_plan(kernel_size, frequencies, uvw, baselines,
                            aterms_offsets, options);
  }
}

std::shared_ptr<Grid> UnifiedOptimized::allocate_grid(size_t nr_w_layers,
                                                      size_t nr_correlations,
                                                      size_t height, size_t width) {
  if (m_use_unified_memory)
  {
    assert(height == width);
    size_t grid_size = height;
    size_t sizeof_grid = nr_w_layers * auxiliary::sizeof_grid(grid_size, nr_correlations);
    InstanceCUDA& device = get_device(0);
    cu::Context& context = device.get_context();
    std::unique_ptr<cu::UnifiedMemory> u_grid(new cu::UnifiedMemory(context, sizeof_grid));
    return std::shared_ptr<Grid>(
        new Grid(std::move(u_grid), nr_w_layers, nr_correlations, height, width));
  } else {
    // Defer call to cpuProxy
    return cpuProxy->allocate_grid(nr_w_layers, nr_correlations, height, width);
  }
}

void UnifiedOptimized::set_grid(std::shared_ptr<Grid> grid) {
  CUDA::set_grid(grid);
  cpuProxy->set_grid(grid);
}

std::shared_ptr<Grid> UnifiedOptimized::get_grid() {
  flush_wtiles();
  return m_grid;
}

void UnifiedOptimized::init_cache(int subgrid_size, float cell_size, float w_step,
                                  const Array1D<float> &shift) {
  // Initialize cache
  Proxy::init_cache(subgrid_size, cell_size, w_step, shift);

  // Initialize wtiles metadata
  m_wtiles = WTiles(m_nr_wtiles, m_wtile_size);

  // Allocate wtiles on GPU
  InstanceCUDA& device = get_device(0);
  int padded_tile_size = m_wtile_size + subgrid_size;
  size_t sizeof_padded_tiles = m_nr_wtiles * NR_CORRELATIONS * padded_tile_size * padded_tile_size * sizeof(idg::float2);
  device.allocate_device_wtiles(sizeof_padded_tiles);

  // Initialize wtiles metadata
  m_wtiles = WTiles(m_nr_wtiles, m_wtile_size);
}

/*
 * W-tiling
 */
void UnifiedOptimized::run_wtiles_to_grid(
      unsigned int subgrid_size,
      float image_size,
      float w_step,
      const Array1D<float>& shift,
      WTileUpdateInfo& wtile_flush_info)
{
  // Load grid
  unsigned int grid_size = m_grid->get_x_dim();

  // Load CUDA objects
  InstanceCUDA& device = get_device(0);
  cu::Context& context = device.get_context();
  cu::Stream& stream = device.get_execute_stream();

  // Load buffers
  cu::DeviceMemory& d_padded_tiles = device.retrieve_device_wtiles();

  // Get information on what wtiles to flush
  unsigned int nr_tiles = wtile_flush_info.wtile_ids.size();
  std::vector<idg::Coordinate>& tile_coordinates = wtile_flush_info.wtile_coordinates;
  std::vector<int>& tile_ids = wtile_flush_info.wtile_ids;
  int padded_tile_size = m_wtile_size + subgrid_size;

  // Compute the maximum w value for all tiles
  float max_abs_w = 0.0;
  for (unsigned int i = 0; i < nr_tiles; i++) {
    idg::Coordinate &coordinate = tile_coordinates[i];
    float w = (coordinate.z + 0.5f) * w_step;
    max_abs_w = std::max(max_abs_w, std::abs(w));
  }

  // Compute the maximum tile size for all padded tiles
  const float image_size_shift =
    image_size + 2 * std::max(std::abs(shift(0)), std::abs(shift(1)));
  int w_padded_tile_size = next_composite(
    padded_tile_size + int(ceil(max_abs_w * image_size_shift * image_size)));

  // Allocate padded tiles buffer
  size_t free_memory = device.get_free_memory();
  size_t sizeof_w_padded_tile = w_padded_tile_size * w_padded_tile_size * NR_CORRELATIONS * sizeof(idg::float2);
  unsigned int nr_tiles_batch = (free_memory * 0.4) / sizeof_w_padded_tile;
  nr_tiles_batch = min(nr_tiles_batch, nr_tiles);
  size_t sizeof_w_padded_tiles = nr_tiles_batch * sizeof_w_padded_tile;
  cu::HostMemory h_w_padded_tiles(context, sizeof_w_padded_tiles);
  cu::DeviceMemory d_w_padded_tiles(context, sizeof_w_padded_tiles);

  // Allocate coordinates buffer
  size_t sizeof_tile_coordinates = nr_tiles_batch * sizeof(idg::Coordinate);
  cu::DeviceMemory d_tile_coordinates(context, sizeof_tile_coordinates);

  // Allocate ids buffer
  size_t sizeof_tile_ids = nr_tiles_batch * sizeof(int);
  cu::DeviceMemory d_tile_ids(context, sizeof_tile_ids);

  // Copy shift to device
  cu::DeviceMemory d_shift(context, shift.bytes());
  stream.memcpyHtoDAsync(d_shift, shift.data(), shift.bytes());

  // Initialize FFT for w_padded_tiles
  unsigned stride = 1;
  unsigned dist = w_padded_tile_size * w_padded_tile_size;
  unsigned batch = nr_tiles_batch * NR_CORRELATIONS;
  cufft::C2C_2D fft(context, w_padded_tile_size, w_padded_tile_size,
                    stride, dist, batch);
  fft.setStream(stream);
  cufftComplex* tile_ptr = reinterpret_cast<cufftComplex*>(static_cast<CUdeviceptr>(d_w_padded_tiles));

  // Iterate all tiles
  double runtime = -omp_get_wtime();
  unsigned int current_nr_tiles = nr_tiles_batch;
  for (unsigned int tile_offset = 0; tile_offset < nr_tiles; tile_offset += current_nr_tiles) {
    current_nr_tiles = tile_offset + current_nr_tiles < nr_tiles
                       ? current_nr_tiles : nr_tiles - tile_offset;
    int tile_id = tile_ids[tile_offset];

    // Copy tile metadata to GPU
    sizeof_tile_ids = current_nr_tiles * sizeof(int);
    stream.memcpyHtoDAsync(d_tile_ids, &tile_ids[tile_offset], sizeof_tile_ids);
    sizeof_tile_coordinates = current_nr_tiles * sizeof(idg::Coordinate);
    stream.memcpyHtoDAsync(d_tile_coordinates, &tile_coordinates[tile_offset], sizeof_tile_coordinates);

    // Call kernel_copy_tile
    device.launch_adder_copy_tiles(
      current_nr_tiles, padded_tile_size, w_padded_tile_size,
      d_tile_ids, d_padded_tiles, d_w_padded_tiles);

    // Launch inverse FFT
    fft.execute(tile_ptr, tile_ptr, CUFFT_INVERSE);

    // Call kernel_apply_phasor
    device.launch_adder_apply_phasor(current_nr_tiles, image_size, w_step, w_padded_tile_size,
      d_w_padded_tiles, d_shift, d_tile_coordinates);

    // Launch forward FFT
    fft.execute(tile_ptr, tile_ptr, CUFFT_FORWARD);

    if (m_use_unified_memory)
    {
      // Call kernel_wtiles_to_grid
      cu::UnifiedMemory u_grid(context, m_grid->data(), m_grid->bytes());
      device.launch_adder_wtiles_to_grid(current_nr_tiles, m_wtile_size,
        w_padded_tile_size, grid_size, d_tile_coordinates, d_w_padded_tiles,
        u_grid);

      // Wait for GPU to finish
      stream.synchronize();
    } else {
      // Copy w_padded_tiles to host
      stream.memcpyDtoHAsync(h_w_padded_tiles, d_w_padded_tiles, sizeof_w_padded_tiles);

      // Wait for GPU to finish
      stream.synchronize();

      // Add tile to grid
      #pragma omp parallel
      {
        for (unsigned tile_index = 0; tile_index < current_nr_tiles; tile_index++) {
          // Compute the padded size of the current tile
          idg::Coordinate &coordinate = tile_coordinates[tile_offset + tile_index];
          float w = (coordinate.z + 0.5f) * w_step;

          int x0 = coordinate.x * m_wtile_size - (w_padded_tile_size - m_wtile_size) / 2 +
                   grid_size / 2;
          int y0 = coordinate.y * m_wtile_size - (w_padded_tile_size - m_wtile_size) / 2 +
                   grid_size / 2;
          int x_start = std::max(0, x0);
          int y_start = std::max(0, y0);
          int x_end = std::min(x0 + w_padded_tile_size, (int) grid_size);
          int y_end = std::min(y0 + w_padded_tile_size, (int) grid_size);

          #pragma omp for
          for (int y = y_start; y < y_end; y++) {
            for (int x = x_start; x < x_end; x++) {
              for (unsigned int pol = 0; pol < NR_CORRELATIONS; pol++) {
                unsigned long dst_idx = index_grid(grid_size, pol, y, x);
                unsigned long src_idx = index_grid(w_padded_tile_size, tile_index, pol, (y - y0), (x - x0));
                std::complex<float>* tile_ptr = (std::complex<float> *) h_w_padded_tiles.ptr();
                std::complex<float>* grid_ptr = m_grid->data();
                grid_ptr[dst_idx] += tile_ptr[src_idx];
              } // end for pol
            } // end for x
          } // end for y
        } // end for tile_index
      } // end omp parallel
    } // end if m_use_unified_memory
  } // end for tile_offset

  runtime += omp_get_wtime();
  std::cout << "runtime: " << runtime << std::endl;
}

void UnifiedOptimized::run_subgrids_to_wtiles(
    unsigned int local_id,
    unsigned int nr_subgrids,
    unsigned int subgrid_size,
    float image_size,
    float w_step,
    const idg::Array1D<float>& shift,
    WTileUpdateSet& wtile_flush_set)
{
  // Load CUDA objects
  InstanceCUDA& device = get_device(0);
  cu::Stream& stream = device.get_execute_stream();

  // Load buffers
  cu::DeviceMemory& d_subgrids = device.retrieve_device_subgrids(local_id);
  cu::DeviceMemory& d_metadata = device.retrieve_device_metadata(local_id);
  cu::DeviceMemory& d_padded_tiles = device.retrieve_device_wtiles();

  for (unsigned int subgrid_index = 0; subgrid_index < nr_subgrids;)
  {
    // Is a flush needed right now?
    if (!wtile_flush_set.empty() && wtile_flush_set.front().subgrid_index ==
                                    (int) (subgrid_index))
    {
      // Get information on what wtiles to flush
      WTileUpdateInfo &wtile_flush_info = wtile_flush_set.front();

      // Project wtiles to master grid
      run_wtiles_to_grid(subgrid_size, image_size, w_step, shift, wtile_flush_info);

      // Remove the flush event from the queue
      wtile_flush_set.pop_front();
    }

    // Initialize number of subgrids to process next to all remaining subgrids
    // in job
    int nr_subgrids_to_process = nr_subgrids - subgrid_index;

    // Check whether a flush needs to happen before the end of the job
    if (!wtile_flush_set.empty() && wtile_flush_set.front().subgrid_index -
                                    (int) subgrid_index < nr_subgrids_to_process)
    {
      // Reduce the number of subgrids to process to just before the next flush
      // event
      nr_subgrids_to_process = wtile_flush_set.front().subgrid_index -
                               subgrid_index;
    }

    // Add all subgrids to the wtiles
    unsigned int grid_size = m_grid->get_x_dim();
    device.launch_adder_subgrids_to_wtiles(nr_subgrids, grid_size, subgrid_size, m_wtile_size,
      subgrid_index, d_metadata, d_subgrids, d_padded_tiles);
    stream.synchronize();

    // Increment the subgrid index by the actual number of processed subgrids
    subgrid_index += nr_subgrids_to_process;
  }
}

void UnifiedOptimized::flush_wtiles() {
  // Get parameters
  unsigned int grid_size      = m_grid->get_x_dim();
  float cell_size             = m_cache_state.cell_size;
  float image_size            = grid_size * cell_size;
  int subgrid_size            = m_cache_state.subgrid_size;
  float w_step                = m_cache_state.w_step;
  const Array1D<float>& shift = m_cache_state.shift;

  // Get all the remaining wtiles
  WTileUpdateInfo wtile_flush_info = m_wtiles.clear();

  // Project wtiles to master grid
  if (wtile_flush_info.wtile_ids.size()) {
    run_wtiles_to_grid(subgrid_size, image_size, w_step, shift, wtile_flush_info);
  }
}

}  // namespace hybrid
}  // namespace proxy
}  // namespace idg
