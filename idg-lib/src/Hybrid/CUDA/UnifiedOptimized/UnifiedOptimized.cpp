// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "UnifiedOptimized.h"

#include <thread>
#include <mutex>
#include <csignal>
#include <algorithm>

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
UnifiedOptimized::UnifiedOptimized()
    : CUDA(default_info()), cpuProxy(new idg::proxy::cpu::Optimized()) {
#if defined(DEBUG)
  std::cout << "UnifiedOptimized::" << __func__ << std::endl;
#endif

  omp_set_nested(true);

  // Increase the fraction of reserved memory
  set_fraction_reserved(0.4);

  initialize_buffers();

  cuProfilerStart();
}

// Destructor
UnifiedOptimized::~UnifiedOptimized() {
#if defined(DEBUG)
  std::cout << "UnifiedOptimized::" << __func__ << std::endl;
#endif

  // Explicitely free the grid before
  // the CUDA Context is destroyed
  if (m_use_unified_memory) {
    m_grid.reset();
  }

  free_buffers();

  cuProfilerStop();
}

void UnifiedOptimized::initialize_buffers() {
  const cu::Context& context = get_device(0).get_context();
  m_buffers_wtiling.d_tiles.reset(new cu::DeviceMemory(context, 0));
  m_buffers_wtiling.d_padded_tiles.reset(new cu::DeviceMemory(context, 0));
  m_buffers_wtiling.h_tiles.reset(new cu::HostMemory(context, 0));
  m_buffers_wtiling.d_patches.resize(m_nr_patches_batch);
  for (unsigned int i = 0; i < m_nr_patches_batch; i++) {
    size_t sizeof_patch = NR_CORRELATIONS * m_patch_size * m_patch_size *
                          sizeof(std::complex<float>);
    m_buffers_wtiling.d_patches[i].reset(
        new cu::DeviceMemory(context, sizeof_patch));
  }
}

void UnifiedOptimized::free_buffers() {
  m_buffers_wtiling.d_tiles.reset();
  m_buffers_wtiling.d_padded_tiles.reset();
  m_buffers_wtiling.h_tiles.reset();
  m_buffers_wtiling.d_patches.clear();
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
  m_report->initialize(nr_channels, subgrid_size, grid_size);
  device.set_report(m_report);
  cpuKernels.set_report(m_report);
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

  // Load memory objects
  cu::DeviceMemory& d_wavenumbers = *m_buffers.d_wavenumbers;
  cu::DeviceMemory& d_spheroidal = *m_buffers.d_spheroidal;
  cu::DeviceMemory& d_aterms = *m_buffers.d_aterms;
  cu::DeviceMemory& d_aterms_indices = *m_buffers.d_aterms_indices_[0];
  cu::DeviceMemory& d_avg_aterm = *m_buffers.d_avg_aterm;

  // Start performance measurement
  startStates[device_id] = device.measure();
  startStates[nr_devices] = hostPowerSensor->read();

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
          *m_buffers.d_visibilities_[local_id_next];
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

    // Initialize subgrids to zero
    d_subgrids.zero(executestream);

    // Wait for input to be copied
    executestream.waitEvent(*inputCopied[job_id]);

    // Launch gridder kernel
    device.launch_gridder(
        current_time_offset, current_nr_subgrids, grid_size, subgrid_size,
        image_size, w_step, nr_channels, nr_stations, shift(0), shift(1), d_uvw,
        d_wavenumbers, d_visibilities, d_spheroidal, d_aterms, d_aterms_indices,
        d_avg_aterm, d_metadata, d_subgrids);

    // Launch FFT
    device.launch_subgrid_fft(d_subgrids, current_nr_subgrids,
                              FourierDomainToImageDomain);

    // Run W-tiling
    auto subgrid_offset = plan.get_subgrid_offset(jobs[job_id].first_bl);
    run_subgrids_to_wtiles(subgrid_offset, current_nr_subgrids, subgrid_size,
                           image_size, w_step, shift, wtile_flush_set,
                           d_subgrids, d_metadata);

    // Report performance
    device.enqueue_report(dtohstream, jobs[job_id].current_nr_timesteps,
                          jobs[job_id].current_nr_subgrids);

    // Wait for GPU to finish
    executestream.record(*gpuFinished[job_id]);
    gpuFinished[job_id]->synchronize();
  }  // end for bl

  // End performance measurement
  endStates[device_id] = device.measure();
  endStates[nr_devices] = hostPowerSensor->read();
  m_report->update(Report::host, startStates[nr_devices],
                   endStates[nr_devices]);

  // Update report
  auto total_nr_subgrids = plan.get_nr_subgrids();
  auto total_nr_timesteps = plan.get_nr_timesteps();
  auto total_nr_visibilities = plan.get_nr_visibilities();
  m_report->print_total(total_nr_timesteps, total_nr_subgrids);
  m_report->print_visibilities(auxiliary::name_gridding, total_nr_visibilities);
}  // end run_gridding

void UnifiedOptimized::do_gridding(
    const Plan& plan, const Array1D<float>& frequencies,
    const Array3D<Visibility<std::complex<float>>>& visibilities,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterms_offsets,
    const Array2D<float>& spheroidal) {
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
  cu::HostMemory& h_subgrids = *m_buffers.h_subgrids;
  h_subgrids.resize(sizeof_subgrids);

  // Performance measurements
  m_report->initialize(nr_channels, subgrid_size, grid_size);
  device.set_report(m_report);
  cpuKernels.set_report(m_report);
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

    // Debug: prevent the use of any pre-existent subgrids on the device
    d_subgrids.zero(executestream);

    // Run W-tiling
    auto subgrid_offset = plan.get_subgrid_offset(jobs[job_id].first_bl);
    run_subgrids_from_wtiles(subgrid_offset, current_nr_subgrids, subgrid_size,
                             image_size, w_step, shift, wtile_initialize_set,
                             d_subgrids, d_metadata);

    // Launch FFT
    device.launch_subgrid_fft(d_subgrids, current_nr_subgrids,
                              ImageDomainToFourierDomain);

    // Launch degridder kernel
    device.launch_degridder(current_time_offset, current_nr_subgrids, grid_size,
                            subgrid_size, image_size, w_step, nr_channels,
                            nr_stations, shift(0), shift(1), d_uvw,
                            d_wavenumbers, d_visibilities, d_spheroidal,
                            d_aterms, d_aterms_indices, d_metadata, d_subgrids);
    executestream.record(*gpuFinished[job_id]);

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

  // Wait for all visibilities to be copied
  dtohstream.synchronize();

  // End performance measurement
  endStates[device_id] = device.measure();
  endStates[nr_devices] = hostPowerSensor->read();
  m_report->update(Report::host, startStates[nr_devices],
                   endStates[nr_devices]);

  // Update report
  auto total_nr_subgrids = plan.get_nr_subgrids();
  auto total_nr_timesteps = plan.get_nr_timesteps();
  auto total_nr_visibilities = plan.get_nr_visibilities();
  m_report->print_total(total_nr_timesteps, total_nr_subgrids);
  m_report->print_visibilities(auxiliary::name_degridding,
                               total_nr_visibilities);
}  // end run_degridding

void UnifiedOptimized::do_degridding(
    const Plan& plan, const Array1D<float>& frequencies,
    Array3D<Visibility<std::complex<float>>>& visibilities,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterms_offsets,
    const Array2D<float>& spheroidal) {
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
    options.w_step = m_cache_state.w_step;
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
                                                      size_t height,
                                                      size_t width) {
  if (m_use_unified_memory) {
    assert(height == width);
    size_t grid_size = height;
    size_t sizeof_grid =
        nr_w_layers * auxiliary::sizeof_grid(grid_size, nr_correlations);
    InstanceCUDA& device = get_device(0);
    cu::Context& context = device.get_context();
    u_grid.reset(new cu::UnifiedMemory(context, sizeof_grid));
    m_grid.reset(
        new Grid(*u_grid, nr_w_layers, nr_correlations, grid_size, grid_size));
    return m_grid;
  } else {
    // Defer call to cpuProxy
    return cpuProxy->allocate_grid(nr_w_layers, nr_correlations, height, width);
  }
}

void UnifiedOptimized::set_grid(std::shared_ptr<Grid> grid) {
  CUDA::set_grid(grid);
  cpuProxy->set_grid(grid);
}

std::shared_ptr<Grid> UnifiedOptimized::get_final_grid() {
  flush_wtiles();
  return m_grid;
}

void UnifiedOptimized::init_cache(int subgrid_size, float cell_size,
                                  float w_step, const Array1D<float>& shift) {
  // Initialize cache
  Proxy::init_cache(subgrid_size, cell_size, w_step, shift);

  // Allocate wtiles on GPU
  InstanceCUDA& device = get_device(0);
  int tile_size = m_tile_size + subgrid_size;

  // Determine the amount of free device memory
  size_t free_memory = device.get_free_memory();

  // Compute the size of one tile
  size_t sizeof_tile =
      NR_CORRELATIONS * tile_size * tile_size * sizeof(idg::float2);

  // We need GPU memory for:
  // - The tiles: d_tiles (tile_size + subgrid_size)
  // - The padded tiles: d_padded_tiles (tile_size + subgrid_size + w_size)
  // - An FFT plan for the padded tiles
  // - Some miscellaneous buffers (tile ids, tile coordinates)
  // Assume that the first three will use the same amount of memory
  // Thus, given that padded tiles are larger than tiles, the padded
  // tiles will always need to be processed in batches.
  m_nr_tiles = (free_memory * 0.25) / sizeof_tile;
  size_t sizeof_tiles = m_nr_tiles * sizeof_tile;
  m_buffers_wtiling.d_tiles->resize(sizeof_tiles);
  m_buffers_wtiling.d_padded_tiles->resize(sizeof_tiles);
  m_buffers_wtiling.h_tiles->resize(sizeof_tiles);

  // Initialize wtiles metadata
  m_wtiles = WTiles(m_nr_tiles, m_tile_size);
}

/*
 * W-tiling
 */
void find_patches_for_tiles(int grid_size, int tile_size, int padded_tile_size,
                            int patch_size, int nr_tiles,
                            const idg::Coordinate* tile_coordinates,
                            std::vector<idg::Coordinate>& patch_coordinates,
                            std::vector<int>& patch_nr_tiles,
                            std::vector<int>& patch_tile_ids,
                            std::vector<int>& patch_tile_id_offsets) {
  // Initialize patch_tile_ids and patch_coordinates
  // for all patches in the grid.
  std::vector<std::vector<int>> patch_tile_ids_;
  std::vector<idg::Coordinate> patch_coordinates_;
  patch_coordinates_.resize(0);

  // Determine the number of patches
  // needed to cover the entire grid.
  int nr_patches = 0;
  for (int y = 0; y < grid_size; y += patch_size) {
    for (int x = 0; x < grid_size; x += patch_size) {
      nr_patches++;
      patch_coordinates_.push_back({x, y});
    }
  }
  patch_tile_ids_.resize(nr_patches);

  // Iterate all patches to find the tiles that (partially) overlap
#pragma omp parallel for
  for (int i = 0; i < nr_patches; i++) {
    patch_tile_ids_[i].resize(0);

    int patch_x_start = patch_coordinates_[i].x;
    int patch_y_start = patch_coordinates_[i].y;
    int patch_x_end = patch_x_start + patch_size;
    int patch_y_end = patch_y_start + patch_size;

    for (int j = 0; j < nr_tiles; j++) {
      // Compute position of tile in grid
      idg::Coordinate coordinate = tile_coordinates[j];
      int x0 = coordinate.x * tile_size - (padded_tile_size - tile_size) / 2 +
               grid_size / 2;
      int y0 = coordinate.y * tile_size - (padded_tile_size - tile_size) / 2 +
               grid_size / 2;
      int x_start = max(0, x0);
      int y_start = max(0, y0);
      int x_end = x_start + padded_tile_size;
      int y_end = y_start + padded_tile_size;

      // Check whether the tile (partially) falls in the patch
      if (!(x_start > patch_x_end || y_start > patch_y_end ||
            x_end < patch_x_start || y_end < patch_y_start)) {
        patch_tile_ids_[i].push_back(j);
      }
    }
  }

  // Reset output vectors
  patch_coordinates.resize(0);
  patch_nr_tiles.resize(0);
  patch_tile_ids.resize(0);
  patch_tile_id_offsets.resize(0);

  // Fill in the output vectors for the non-empty patches
  for (int i = 0; i < nr_patches; i++) {
    int nr_tiles_patch = patch_tile_ids_[i].size();

    if (nr_tiles_patch > 0) {
      // Add the number of tiles for the current patch
      patch_nr_tiles.push_back(nr_tiles_patch);

      // Add the coordinate for the current patch
      patch_coordinates.push_back(patch_coordinates_[i]);

      // Add the tile id offset to the number of tile ids
      // in all previous non-empty patches.
      patch_tile_id_offsets.push_back(patch_tile_ids.size());

      // Add all tile ids for the current patch
      for (int j = 0; j < nr_tiles_patch; j++) {
        patch_tile_ids.push_back(patch_tile_ids_[i][j]);
      }
    }
  }
}

void run_adder_patch_to_grid(int grid_size, int patch_size, int nr_patches,
                             idg::Coordinate* __restrict__ patch_coordinates,
                             std::complex<float>* __restrict__ grid,
                             std::complex<float>* __restrict__ patches_buffer) {
  std::complex<float>* dst_ptr = grid;
  std::complex<float>* src_ptr = patches_buffer;

#pragma omp parallel for
  for (int y_ = 0; y_ < patch_size; y_++) {
    for (int i = 0; i < nr_patches; i++) {
      int x = patch_coordinates[i].x;
      int y = patch_coordinates[i].y;
      for (int pol = 0; pol < NR_CORRELATIONS; pol++) {
        for (int x_ = 0; x_ < patch_size; x_++) {
          size_t dst_idx = index_grid(grid_size, pol, y + y_, x + x_);
          size_t src_idx = index_grid(patch_size, i, pol, y_, x_);
          dst_ptr[dst_idx] += src_ptr[src_idx];
        }  // end for x_
      }    // end for pol
    }      // end for i
  }        // end for y_
}

void UnifiedOptimized::run_wtiles_to_grid(unsigned int subgrid_size,
                                          float image_size, float w_step,
                                          const Array1D<float>& shift,
                                          WTileUpdateInfo& wtile_flush_info) {
  // Load grid
  unsigned int grid_size = m_grid->get_x_dim();

  // Load CUDA objects
  InstanceCUDA& device = get_device(0);
  cu::Context& context = device.get_context();
  cu::Stream& executestream = device.get_execute_stream();
  cu::Stream& dtohstream = device.get_dtoh_stream();

  // Load buffers
  cu::DeviceMemory& d_tiles = *m_buffers_wtiling.d_tiles;
  cu::DeviceMemory& d_padded_tiles = *m_buffers_wtiling.d_padded_tiles;
  cu::HostMemory& h_padded_tiles = *m_buffers_wtiling.h_tiles;

  // Get information on what wtiles to flush
  const int tile_size = m_tile_size;
  const unsigned int nr_tiles = wtile_flush_info.wtile_ids.size();
  std::vector<idg::Coordinate>& tile_coordinates =
      wtile_flush_info.wtile_coordinates;
  std::vector<int>& tile_ids = wtile_flush_info.wtile_ids;
  const int padded_tile_size = tile_size + subgrid_size;

  // Compute w_padded_tile_size for all tiles
  const float image_size_shift =
      image_size + 2 * std::max(std::abs(shift(0)), std::abs(shift(1)));
  std::vector<int> w_padded_tile_sizes = compute_w_padded_tile_sizes(
      tile_coordinates.data(), nr_tiles, w_step, image_size, image_size_shift,
      padded_tile_size);

  // Find the maximum tile size for all padded tiles
  int w_padded_tile_size =
      *std::max_element(w_padded_tile_sizes.begin(), w_padded_tile_sizes.end());

  std::cout << "tile_size: " << tile_size << std::endl;
  std::cout << "padded_tile_size: " << padded_tile_size << std::endl;
  std::cout << "w_padded_tile_size: " << w_padded_tile_size << std::endl;

  // Compute the number of padded tiles
  size_t sizeof_w_padded_tile = w_padded_tile_size * w_padded_tile_size *
                                NR_CORRELATIONS * sizeof(idg::float2);
  unsigned int nr_tiles_batch =
      (d_padded_tiles.size() / sizeof_w_padded_tile) / 2;
  nr_tiles_batch = min(nr_tiles_batch, nr_tiles);

  // Allocate coordinates buffer
  size_t sizeof_tile_coordinates = nr_tiles_batch * sizeof(idg::Coordinate);
  cu::DeviceMemory d_tile_coordinates(context, sizeof_tile_coordinates);

  // Allocate ids buffer
  size_t sizeof_tile_ids = nr_tiles_batch * sizeof(int);
  cu::DeviceMemory d_tile_ids(context, sizeof_tile_ids);
  cu::DeviceMemory d_padded_tile_ids(context, sizeof_tile_ids);
  cu::DeviceMemory d_packed_tile_ids(context, sizeof_tile_ids);

  // Initialize d_padded_tile_ids
  std::vector<int> padded_tile_ids(nr_tiles_batch);
  for (unsigned int i = 0; i < nr_tiles_batch; i++) {
    padded_tile_ids[i] = i;
  }

  // Copy shift to device
  cu::DeviceMemory d_shift(context, shift.bytes());
  executestream.memcpyHtoDAsync(d_shift, shift.data(), shift.bytes());

  // Create jobs
  struct JobData {
    int tile_offset;
    int current_nr_tiles;
  };

  std::vector<JobData> jobs;

  unsigned int current_nr_tiles = nr_tiles_batch;
  for (unsigned int tile_offset = 0; tile_offset < nr_tiles;
       tile_offset += current_nr_tiles) {
    current_nr_tiles = std::min(current_nr_tiles, nr_tiles - tile_offset);

    JobData job;
    job.current_nr_tiles = current_nr_tiles;
    job.tile_offset = tile_offset;
    jobs.push_back(std::move(job));
  }

  // FFT plan
  std::unique_ptr<cufft::C2C_2D> fft;

  // Iterate all jobs
  int last_w_padded_tile_size = w_padded_tile_size;
  for (auto& job : jobs) {
    int tile_offset = job.tile_offset;
    int current_nr_tiles = job.current_nr_tiles;

    // Set w_padded_tile_size for current job
    int current_w_padded_tile_size = *std::max_element(
        w_padded_tile_sizes.begin() + tile_offset,
        w_padded_tile_sizes.begin() + tile_offset + current_nr_tiles);

    cufftComplex* tile_ptr = reinterpret_cast<cufftComplex*>(
        static_cast<CUdeviceptr>(d_padded_tiles));

    if (tile_offset == 0 ||
        current_w_padded_tile_size != last_w_padded_tile_size)
    {
      // Initialize FFT for w_padded_tiles
      unsigned stride = 1;
      unsigned dist = current_w_padded_tile_size * current_w_padded_tile_size;
      unsigned batch = nr_tiles_batch * NR_CORRELATIONS;

      fft.reset(new cufft::C2C_2D(context, current_w_padded_tile_size,
        current_w_padded_tile_size, stride, dist, batch));
      fft->setStream(executestream);

      last_w_padded_tile_size = current_w_padded_tile_size;
    }

    // Copy tile metadata to GPU
    sizeof_tile_ids = current_nr_tiles * sizeof(int);
    executestream.memcpyHtoDAsync(d_tile_ids, &tile_ids[tile_offset],
                                  sizeof_tile_ids);
    executestream.memcpyHtoDAsync(d_padded_tile_ids, padded_tile_ids.data(),
                                  sizeof_tile_ids);
    sizeof_tile_coordinates = current_nr_tiles * sizeof(idg::Coordinate);
    executestream.memcpyHtoDAsync(d_tile_coordinates,
                                  &tile_coordinates[tile_offset],
                                  sizeof_tile_coordinates);

    // Call kernel_copy_tiles
    device.launch_copy_tiles(current_nr_tiles, padded_tile_size,
                             current_w_padded_tile_size, d_tile_ids, d_padded_tile_ids,
                             d_tiles, d_padded_tiles);

    // Launch inverse FFT
    fft->execute(tile_ptr, tile_ptr, CUFFT_INVERSE);

    // Call kernel_apply_phasor
    device.launch_apply_phasor_to_wtiles(current_nr_tiles, image_size, w_step,
                                         current_w_padded_tile_size, d_padded_tiles,
                                         d_shift, d_tile_coordinates, -1);

    // Launch forward FFT
    fft->execute(tile_ptr, tile_ptr, CUFFT_FORWARD);

    if (m_use_unified_memory) {
      // Call kernel_wtiles_to_grid
      cu::UnifiedMemory u_grid(context, m_grid->data(), m_grid->bytes());
      device.launch_adder_wtiles_to_grid(
          current_nr_tiles, grid_size, tile_size, current_w_padded_tile_size,
          d_padded_tile_ids, d_tile_coordinates, d_padded_tiles, u_grid);

      // Wait for GPU to finish
      executestream.synchronize();
    } else {
      // Wait for GPU to finish
      executestream.synchronize();

      // Find all tiles that (partially) fit in the current patch
      std::vector<idg::Coordinate> patch_coordinates;
      std::vector<int> patch_nr_tiles;
      std::vector<int> patch_tile_ids;
      std::vector<int> patch_tile_id_offsets;
      find_patches_for_tiles(
          grid_size, tile_size, current_w_padded_tile_size, m_patch_size,
          current_nr_tiles, &tile_coordinates[tile_offset], patch_coordinates,
          patch_nr_tiles, patch_tile_ids, patch_tile_id_offsets);
      unsigned int total_nr_patches = patch_coordinates.size();

      // Iterate patches in batches (note: reusing h_padded_tiles for patches)
      size_t sizeof_patch = m_buffers_wtiling.d_patches[0]->size();
      unsigned int max_nr_patches = h_padded_tiles.size() / sizeof_patch;
      unsigned int current_nr_patches = max_nr_patches;

      // Events
      std::vector<std::unique_ptr<cu::Event>> gpuFinished;
      std::vector<std::unique_ptr<cu::Event>> outputCopied;
      for (unsigned int i = 0; i < m_nr_patches_batch; i++) {
        gpuFinished.emplace_back(new cu::Event(context));
        outputCopied.emplace_back(new cu::Event(context));
      }

      for (unsigned int patch_offset = 0; patch_offset < total_nr_patches;
           patch_offset += current_nr_patches) {
        for (unsigned int i = 0; i < current_nr_patches; i++) {
          current_nr_patches =
              min(current_nr_patches, total_nr_patches - patch_offset);

          int id = i % m_nr_patches_batch;
          cu::DeviceMemory& d_patch = *(m_buffers_wtiling.d_patches[id]);

          // Wait for previous patch to be computed
          if (i > m_nr_patches_batch) {
            gpuFinished[id]->synchronize();
          }

          // Get patch metadata
          int patch_id = patch_offset + i;
          int* packed_tile_ids =
              &patch_tile_ids[patch_tile_id_offsets[patch_id]];
          idg::Coordinate patch_coordinate = patch_coordinates[patch_id];
          int current_nr_tiles = patch_nr_tiles[patch_id];

          // Copy packed tile ids to GPU
          sizeof_tile_ids = current_nr_tiles * sizeof(int);
          executestream.memcpyHtoDAsync(d_packed_tile_ids, packed_tile_ids,
                                        sizeof_tile_ids);

          // Reset patch
          executestream.waitEvent(*outputCopied[id]);
          d_patch.zero(executestream);

          // Combine tiles onto patch
          device.launch_adder_wtiles_to_patch(
              current_nr_tiles, grid_size, padded_tile_size - subgrid_size,
              current_w_padded_tile_size, m_patch_size, patch_coordinate,
              d_packed_tile_ids, d_tile_coordinates, d_padded_tiles, d_patch);
          executestream.record(*gpuFinished[id]);

          // Copy patch to the host
          void* patch_ptr =
              static_cast<char*>(h_padded_tiles) + patch_id * sizeof_patch;
          dtohstream.waitEvent(*gpuFinished[id]);
          dtohstream.memcpyDtoHAsync(patch_ptr, d_patch, sizeof_patch);
          dtohstream.record(*outputCopied[id]);
        }

        // Wait for patches to be copied
        dtohstream.synchronize();

        // Add patch to the grid
        cu::Marker marker("patch_to_grid", cu::Marker::red);
        marker.start();

        run_adder_patch_to_grid(grid_size, m_patch_size, current_nr_patches,
                                patch_coordinates.data(), m_grid->data(),
                                h_padded_tiles);
        marker.end();
      }  // end for patch_offset
    }  // end if m_use_unified_memory
  }    // end for jobs
}

void UnifiedOptimized::run_subgrids_to_wtiles(
    unsigned int subgrid_offset, unsigned int nr_subgrids,
    unsigned int subgrid_size, float image_size, float w_step,
    const idg::Array1D<float>& shift, WTileUpdateSet& wtile_flush_set,
    cu::DeviceMemory& d_subgrids, cu::DeviceMemory& d_metadata) {
  // Load CUDA objects
  InstanceCUDA& device = get_device(0);
  cu::Stream& stream = device.get_execute_stream();
  cu::DeviceMemory& d_tiles = *m_buffers_wtiling.d_tiles;

  // Performance measurement
  State startState, endState;
  startState = device.measure();

  for (unsigned int subgrid_index = 0; subgrid_index < nr_subgrids;) {
    // Is a flush needed right now?
    if (!wtile_flush_set.empty() && wtile_flush_set.front().subgrid_index ==
                                        (int)(subgrid_index + subgrid_offset)) {
      // Get information on what wtiles to flush
      WTileUpdateInfo& wtile_flush_info = wtile_flush_set.front();

      // Project wtiles to master grid
      run_wtiles_to_grid(subgrid_size, image_size, w_step, shift,
                         wtile_flush_info);

      // Remove the flush event from the queue
      wtile_flush_set.pop_front();
    }

    // Initialize number of subgrids to process next to all remaining subgrids
    // in job
    int nr_subgrids_to_process = nr_subgrids - subgrid_index;

    // Check whether a flush needs to happen before the end of the job
    if (!wtile_flush_set.empty() &&
        wtile_flush_set.front().subgrid_index -
                (int)(subgrid_index + subgrid_offset) <
            nr_subgrids_to_process) {
      // Reduce the number of subgrids to process to just before the next flush
      // event
      nr_subgrids_to_process = wtile_flush_set.front().subgrid_index -
                               (subgrid_index + subgrid_offset);
    }

    // Add all subgrids to the wtiles
    unsigned int grid_size = m_grid->get_x_dim();
    int N = subgrid_size * subgrid_size;
    std::complex<float> scale(1.0f / N, 1.0f / N);
    device.launch_adder_subgrids_to_wtiles(
        nr_subgrids_to_process, grid_size, subgrid_size, m_tile_size,
        subgrid_index, d_metadata, d_subgrids, d_tiles, scale);
    stream.synchronize();

    // Increment the subgrid index by the actual number of processed subgrids
    subgrid_index += nr_subgrids_to_process;
  }

  // End performance measurement
  endState = device.measure();
  m_report->update(Report::wtiling_forward, startState, endState);
}

void UnifiedOptimized::run_wtiles_from_grid(
    unsigned int subgrid_size, float image_size, float w_step,
    const Array1D<float>& shift, WTileUpdateInfo& wtile_initialize_info) {
  // Load grid
  unsigned int grid_size = m_grid->get_x_dim();

  // Load CUDA objects
  InstanceCUDA& device = get_device(0);
  cu::Context& context = device.get_context();
  cu::Stream& executestream = device.get_execute_stream();
  cu::Stream& htodstream = device.get_htod_stream();

  // Load CPU object
  InstanceCPU& cpuKernels = cpuProxy->get_kernels();

  // Load buffers
  cu::DeviceMemory& d_tiles = *m_buffers_wtiling.d_tiles;
  cu::DeviceMemory& d_padded_tiles = *m_buffers_wtiling.d_padded_tiles;
  cu::HostMemory& h_padded_tiles = *m_buffers_wtiling.h_tiles;

  // Get information on what wtiles to flush
  const int tile_size = m_tile_size;
  const unsigned int nr_tiles = wtile_initialize_info.wtile_ids.size();
  std::vector<idg::Coordinate>& tile_coordinates =
      wtile_initialize_info.wtile_coordinates;
  std::vector<int>& tile_ids = wtile_initialize_info.wtile_ids;

  // Compute w_padded_tile_size for all tiles
  const int padded_tile_size = tile_size + subgrid_size;
  const float image_size_shift =
      image_size + 2 * std::max(std::abs(shift(0)), std::abs(shift(1)));
  std::vector<int> w_padded_tile_sizes = compute_w_padded_tile_sizes(
      tile_coordinates.data(), nr_tiles, w_step, image_size, image_size_shift,
      padded_tile_size);

  // Find the maximum tile size for all padded tiles
  int w_padded_tile_size =
      *std::max_element(w_padded_tile_sizes.begin(), w_padded_tile_sizes.end());

  std::cout << "tile_size: " << tile_size << std::endl;
  std::cout << "padded_tile_size: " << padded_tile_size << std::endl;
  std::cout << "w_padded_tile_size: " << w_padded_tile_size << std::endl;

  // Compute the number of padded tiles
  size_t sizeof_w_padded_tile = w_padded_tile_size * w_padded_tile_size *
                                NR_CORRELATIONS * sizeof(idg::float2);
  unsigned int nr_tiles_batch =
      (d_padded_tiles.size() / sizeof_w_padded_tile) / 2;
  nr_tiles_batch = min(nr_tiles_batch, nr_tiles);

  // Allocate coordinates buffer
  size_t sizeof_tile_coordinates = nr_tiles_batch * sizeof(idg::Coordinate);
  cu::DeviceMemory d_tile_coordinates(context, sizeof_tile_coordinates);

  // Allocate ids buffer
  size_t sizeof_tile_ids = nr_tiles_batch * sizeof(int);
  cu::DeviceMemory d_tile_ids(context, sizeof_tile_ids);
  cu::DeviceMemory d_padded_tile_ids(context, sizeof_tile_ids);

  // Initialize d_padded_tile_ids
  std::vector<int> padded_tile_ids(nr_tiles_batch);
  for (unsigned int i = 0; i < nr_tiles_batch; i++) {
    padded_tile_ids[i] = i;
  }
  executestream.memcpyHtoDAsync(d_padded_tile_ids, padded_tile_ids.data(),
                                sizeof_tile_ids);

  // Copy shift to device
  cu::DeviceMemory d_shift(context, shift.bytes());
  executestream.memcpyHtoDAsync(d_shift, shift.data(), shift.bytes());

  // Initialize FFT for w_padded_tiles
  unsigned stride = 1;
  unsigned dist = w_padded_tile_size * w_padded_tile_size;
  unsigned batch = nr_tiles_batch * NR_CORRELATIONS;
  cufft::C2C_2D fft(context, w_padded_tile_size, w_padded_tile_size, stride,
                    dist, batch);
  fft.setStream(executestream);
  cufftComplex* tile_ptr =
      reinterpret_cast<cufftComplex*>(static_cast<CUdeviceptr>(d_padded_tiles));

  // Create jobs
  struct JobData {
    int tile_offset;
    int current_nr_tiles;
  };

  std::vector<JobData> jobs;

  unsigned int current_nr_tiles = nr_tiles_batch;
  for (unsigned int tile_offset = 0; tile_offset < nr_tiles;
       tile_offset += current_nr_tiles) {
    current_nr_tiles = std::min(current_nr_tiles, nr_tiles - tile_offset);

    JobData job;
    job.current_nr_tiles = current_nr_tiles;
    job.tile_offset = tile_offset;
    jobs.push_back(std::move(job));
  }

  // Iterate all jobs
  for (auto& job : jobs) {
    int tile_offset = job.tile_offset;
    int current_nr_tiles = job.current_nr_tiles;

    // Copy tile metadata to GPU
    sizeof_tile_ids = current_nr_tiles * sizeof(int);
    executestream.memcpyHtoDAsync(d_tile_ids, &tile_ids[tile_offset],
                                  sizeof_tile_ids);
    sizeof_tile_coordinates = current_nr_tiles * sizeof(idg::Coordinate);
    executestream.memcpyHtoDAsync(d_tile_coordinates,
                                  &tile_coordinates[tile_offset],
                                  sizeof_tile_coordinates);

    // Split tile from grid
    if (m_use_unified_memory) {
      cu::UnifiedMemory u_grid(context, m_grid->data(), m_grid->bytes());
      device.launch_splitter_wtiles_from_grid(
          current_nr_tiles, grid_size, tile_size, w_padded_tile_size,
          d_padded_tile_ids, d_tile_coordinates, d_padded_tiles, u_grid);
    } else {
      h_padded_tiles.zero();
      std::complex<float>* tile_ptr =
          static_cast<std::complex<float>*>(h_padded_tiles.ptr());
      cpuKernels.run_splitter_wtiles_to_grid(
          current_nr_tiles, tile_size, w_padded_tile_size, grid_size,
          &tile_coordinates[tile_offset], tile_ptr, m_grid->data());

      // Copy tiles to GPU
      size_t sizeof_copy = current_nr_tiles * sizeof_w_padded_tile;
      htodstream.memcpyHtoDAsync(d_padded_tiles, h_padded_tiles, sizeof_copy);

      // Wait for all tiles to be copied
      htodstream.synchronize();
    }

    // Launch inverse FFT
    fft.execute(tile_ptr, tile_ptr, CUFFT_INVERSE);

    // Call kernel_apply_phasor
    device.launch_apply_phasor_to_wtiles(current_nr_tiles, image_size, w_step,
                                         w_padded_tile_size, d_padded_tiles,
                                         d_shift, d_tile_coordinates, 1);

    // Launch forward FFT
    fft.execute(tile_ptr, tile_ptr, CUFFT_FORWARD);

    // Call kernel_copy_tiles
    device.launch_copy_tiles(current_nr_tiles, w_padded_tile_size,
                             padded_tile_size, d_padded_tile_ids, d_tile_ids,
                             d_padded_tiles, d_tiles);
  }  // end for tile_offset
}

void UnifiedOptimized::run_subgrids_from_wtiles(
    unsigned int subgrid_offset, unsigned int nr_subgrids,
    unsigned int subgrid_size, float image_size, float w_step,
    const Array1D<float>& shift, WTileUpdateSet& wtile_initialize_set,
    cu::DeviceMemory& d_subgrids, cu::DeviceMemory& d_metadata) {
  // Load CUDA objects
  InstanceCUDA& device = get_device(0);
  cu::Stream& stream = device.get_execute_stream();

  // Load buffers
  cu::DeviceMemory& d_tiles = *m_buffers_wtiling.d_tiles;

  // Performance measurement
  State startState, endState;
  startState = device.measure();

  for (unsigned int subgrid_index = 0; subgrid_index < nr_subgrids;) {
    // Check whether initialize is needed right now
    if (!wtile_initialize_set.empty() &&
        wtile_initialize_set.front().subgrid_index ==
            (int)(subgrid_index + subgrid_offset)) {
      // Get information on what wtiles to initialize
      WTileUpdateInfo& wtile_initialize_info = wtile_initialize_set.front();

      // Initialize the wtiles from the grid
      run_wtiles_from_grid(subgrid_size, image_size, w_step, shift,
                           wtile_initialize_info);

      // Remove the initialize event from the queue
      wtile_initialize_set.pop_front();
    }

    // Initialize number of subgrids to process next to all remaining subgrids
    // in job
    int nr_subgrids_to_process = nr_subgrids - subgrid_index;

    // Check whether initialization needs to happen before the end of the job
    if (!wtile_initialize_set.empty() &&
        wtile_initialize_set.front().subgrid_index -
                (int)(subgrid_index - subgrid_offset) <
            nr_subgrids_to_process) {
      // Reduce the number of subgrids to process to just before the next
      // initialization event
      nr_subgrids_to_process = wtile_initialize_set.front().subgrid_index -
                               (subgrid_offset + subgrid_index);
    }

    // Process all subgrids that can be processed now
    std::complex<float> scale(1.0f, 1.0f);
    unsigned int grid_size = m_grid->get_x_dim();
    device.launch_splitter_subgrids_from_wtiles(
        nr_subgrids_to_process, grid_size, subgrid_size, m_tile_size,
        subgrid_index, d_metadata, d_subgrids, d_tiles, scale);  // scale?
    stream.synchronize();

    // Increment the subgrid index by the actual number of processed subgrids
    subgrid_index += nr_subgrids_to_process;
  }

  // End performance measurement
  endState = device.measure();
  m_report->update(Report::wtiling_backward, startState, endState);
}

void UnifiedOptimized::flush_wtiles() {
  // Get parameters
  unsigned int grid_size = m_grid->get_x_dim();
  float cell_size = m_cache_state.cell_size;
  float image_size = grid_size * cell_size;
  int subgrid_size = m_cache_state.subgrid_size;
  float w_step = m_cache_state.w_step;
  const Array1D<float>& shift = m_cache_state.shift;

  // Get all the remaining wtiles
  WTileUpdateInfo wtile_flush_info = m_wtiles.clear();

  // Project wtiles to master grid
  if (wtile_flush_info.wtile_ids.size()) {
    m_report->initialize();
    InstanceCUDA& device = get_device(0);
    State startState, endState;
    startState = device.measure();
    run_wtiles_to_grid(subgrid_size, image_size, w_step, shift,
                       wtile_flush_info);
    endState = device.measure();
    m_report->update(Report::wtiling_forward, startState, endState);
    m_report->print_total();
  }
}

}  // namespace hybrid
}  // namespace proxy
}  // namespace idg
