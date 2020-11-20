// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>  // max_element

#include "Generic.h"
#include "InstanceCUDA.h"

using namespace idg::kernel::cuda;
using namespace powersensor;

namespace idg {
namespace proxy {
namespace cuda {

// Constructor
Generic::Generic(ProxyInfo info) : CUDA(info) {
#if defined(DEBUG)
  std::cout << "Generic::" << __func__ << std::endl;
#endif

  // Initialize host PowerSensor
  hostPowerSensor = get_power_sensor(sensor_host);
}

// Destructor
Generic::~Generic() { delete hostPowerSensor; }

/* High level routines */
void Generic::do_transform(DomainAtoDomainB direction,
                           Array3D<std::complex<float>>& grid) {
#if defined(DEBUG)
  std::cout << __func__ << std::endl;
  std::cout << "Transform direction: " << direction << std::endl;
#endif

  // Constants
  auto grid_size = grid.get_x_dim();

  // Load device
  InstanceCUDA& device = get_device(0);

  // Initialize
  cu::Stream& stream = device.get_execute_stream();
  device.set_context();

  // Device memory
  cu::DeviceMemory& d_grid = device.retrieve_device_grid();

  // Performance measurements
  report.initialize(0, 0, grid_size);
  device.set_report(report);
  PowerRecord powerRecords[4];
  State powerStates[4];
  powerStates[0] = hostPowerSensor->read();
  powerStates[2] = device.measure();

  // Perform fft shift
  device.shift(grid);

  // Copy grid to device
  device.measure(powerRecords[0], stream);
  device.copy_htod(stream, d_grid, grid.data(), grid.bytes());
  device.measure(powerRecords[1], stream);

  // Execute fft
  device.launch_grid_fft(d_grid, grid_size, direction);

  // Copy grid to host
  device.measure(powerRecords[2], stream);
  device.copy_dtoh(stream, grid.data(), d_grid, grid.bytes());
  device.measure(powerRecords[3], stream);
  stream.synchronize();

  // Perform fft shift
  device.shift(grid);

  // Perform fft scaling
  std::complex<float> scale =
      std::complex<float>(2.0 / (grid_size * grid_size), 0);
  if (direction == FourierDomainToImageDomain) {
    device.scale(grid, scale);
  }

  // End measurements
  stream.synchronize();
  powerStates[1] = hostPowerSensor->read();
  powerStates[3] = device.measure();

  // Report performance
  report.update_input(powerRecords[0].state, powerRecords[1].state);
  report.update_output(powerRecords[2].state, powerRecords[3].state);
  report.update_host(powerStates[0], powerStates[1]);
  report.print_total();
  report.print_device(powerRecords[0].state, powerRecords[3].state);
}  // end transform

void Generic::run_gridding(
    const Plan& plan, const float w_step, const Array1D<float>& shift,
    const float cell_size, const unsigned int kernel_size,
    const unsigned int subgrid_size, const Array1D<float>& frequencies,
    const Array3D<Visibility<std::complex<float>>>& visibilities,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines, Grid& grid,
    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterms_offsets,
    const Array2D<float>& spheroidal) {
#if defined(DEBUG)
  std::cout << "Generic::" << __func__ << std::endl;
#endif

  InstanceCUDA& device = get_device(0);
  device.set_context();

  // Arguments
  auto nr_baselines = visibilities.get_z_dim();
  auto nr_timesteps = visibilities.get_y_dim();
  auto nr_channels = visibilities.get_x_dim();
  auto nr_stations = aterms.get_z_dim();
  auto grid_size = grid.get_x_dim();
  auto image_size = cell_size * grid_size;

  // Configuration
  const unsigned nr_devices = get_num_devices();
  int device_id = 0;  // only one GPU is used
  int jobsize = m_gridding_state.jobsize[device_id];

  // Page-locked host memory
  device.register_host_memory((void*)plan.get_metadata_ptr(),
                              plan.get_sizeof_metadata());

  // Performance measurements
  report.initialize(nr_channels, subgrid_size, grid_size);
  device.set_report(report);
  std::vector<State> startStates(nr_devices + 1);
  std::vector<State> endStates(nr_devices + 1);

  // Events
  std::vector<std::unique_ptr<cu::Event>> inputCopied;
  std::vector<std::unique_ptr<cu::Event>> gpuFinished;
  for (unsigned bl = 0; bl < nr_baselines; bl += jobsize) {
    inputCopied.push_back(
        std::unique_ptr<cu::Event>(new cu::Event(CU_EVENT_BLOCKING_SYNC)));
    gpuFinished.push_back(
        std::unique_ptr<cu::Event>(new cu::Event(CU_EVENT_BLOCKING_SYNC)));
  }

  // Load memory objects
  cu::DeviceMemory& d_wavenumbers = device.retrieve_device_wavenumbers();
  cu::DeviceMemory& d_spheroidal = device.retrieve_device_spheroidal();
  cu::DeviceMemory& d_aterms = device.retrieve_device_aterms();
  cu::DeviceMemory& d_aterms_indices = device.retrieve_device_aterms_indices();
  cu::DeviceMemory& d_avg_aterm_correction =
      device.retrieve_device_avg_aterm_correction();

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
      executestream.waitEvent(*gpuFinished[job_id - 2]);
    }

    // Initialize subgrids to zero
    d_subgrids.zero(executestream);

    // Wait for input to be copied
    executestream.waitEvent(*inputCopied[job_id]);

    // Launch gridder kernel
    device.launch_gridder(current_time_offset, current_nr_subgrids, grid_size,
                          subgrid_size, image_size, w_step, nr_channels,
                          nr_stations, d_uvw, d_wavenumbers, d_visibilities,
                          d_spheroidal, d_aterms, d_aterms_indices,
                          d_avg_aterm_correction, d_metadata, d_subgrids);

    // Launch FFT
    device.launch_subgrid_fft(d_subgrids, current_nr_subgrids,
                              FourierDomainToImageDomain);

    // Launch adder kernel
    if (m_use_unified_memory) {
      device.launch_adder_unified(current_nr_subgrids, grid_size, subgrid_size,
                                  d_metadata, d_subgrids, grid.data());
    } else {
      cu::DeviceMemory& d_grid = device.retrieve_device_grid();
      device.launch_adder(current_nr_subgrids, grid_size, subgrid_size,
                          d_metadata, d_subgrids, d_grid);
    }
    executestream.record(*gpuFinished[job_id]);

    // Report performance
    device.enqueue_report(executestream, jobs[job_id].current_nr_timesteps,
                          jobs[job_id].current_nr_subgrids);

    // Wait for adder to finish
    gpuFinished[job_id]->synchronize();
  }  // end for bl

  // Wait for all reports to be printed
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
  report.print_visibilities(auxiliary::name_gridding, total_nr_visibilities);
}  // end run_gridding

void Generic::do_gridding(
    const Plan& plan,
    const float w_step,  // in lambda
    const Array1D<float>& shift, const float cell_size,
    const unsigned int kernel_size,  // full width in pixels
    const unsigned int subgrid_size, const Array1D<float>& frequencies,
    const Array3D<Visibility<std::complex<float>>>& visibilities,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines, Grid& grid,
    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterms_offsets,
    const Array2D<float>& spheroidal) {
#if defined(DEBUG)
  std::cout << "Generic::" << __func__ << std::endl;
#endif

#if defined(DEBUG)
  std::clog << "### Initialize gridding" << std::endl;
#endif
  CUDA::initialize(plan, w_step, shift, cell_size, kernel_size, subgrid_size,
                   frequencies, visibilities, uvw, baselines, aterms,
                   aterms_offsets, spheroidal);

#if defined(DEBUG)
  std::clog << "### Run gridding" << std::endl;
#endif
  run_gridding(plan, w_step, shift, cell_size, kernel_size, subgrid_size,
               frequencies, visibilities, uvw, baselines, grid, aterms,
               aterms_offsets, spheroidal);

#if defined(DEBUG)
  std::clog << "### Finish gridding" << std::endl;
#endif
}

void Generic::run_degridding(
    const Plan& plan, const float w_step, const Array1D<float>& shift,
    const float cell_size, const unsigned int kernel_size,
    const unsigned int subgrid_size, const Array1D<float>& frequencies,
    Array3D<Visibility<std::complex<float>>>& visibilities,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Grid& grid, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterms_offsets,
    const Array2D<float>& spheroidal) {
#if defined(DEBUG)
  std::cout << "Generic::" << __func__ << std::endl;
#endif

  InstanceCUDA& device = get_device(0);
  device.set_context();

  // Arguments
  auto nr_baselines = visibilities.get_z_dim();
  auto nr_timesteps = visibilities.get_y_dim();
  auto nr_channels = visibilities.get_x_dim();
  auto nr_stations = aterms.get_z_dim();
  auto grid_size = grid.get_x_dim();
  auto image_size = cell_size * grid_size;

  // Configuration
  const unsigned nr_devices = get_num_devices();
  int device_id = 0;  // only one GPU is used
  int jobsize = m_gridding_state.jobsize[device_id];

  // Page-locked host memory
  device.register_host_memory((void*)plan.get_metadata_ptr(),
                              plan.get_sizeof_metadata());

  // Performance measurements
  report.initialize(nr_channels, subgrid_size, grid_size);
  device.set_report(report);
  std::vector<State> startStates(nr_devices + 1);
  std::vector<State> endStates(nr_devices + 1);

  // Events
  std::vector<std::unique_ptr<cu::Event>> inputCopied;
  std::vector<std::unique_ptr<cu::Event>> gpuFinished;
  std::vector<std::unique_ptr<cu::Event>> outputCopied;
  for (unsigned bl = 0; bl < nr_baselines; bl += jobsize) {
    inputCopied.push_back(
        std::unique_ptr<cu::Event>(new cu::Event(CU_EVENT_BLOCKING_SYNC)));
    gpuFinished.push_back(
        std::unique_ptr<cu::Event>(new cu::Event(CU_EVENT_BLOCKING_SYNC)));
    outputCopied.push_back(
        std::unique_ptr<cu::Event>(new cu::Event(CU_EVENT_BLOCKING_SYNC)));
  }

  // Load memory objects
  cu::DeviceMemory& d_wavenumbers = device.retrieve_device_wavenumbers();
  cu::DeviceMemory& d_spheroidal = device.retrieve_device_spheroidal();
  cu::DeviceMemory& d_aterms = device.retrieve_device_aterms();
  cu::DeviceMemory& d_aterms_indices = device.retrieve_device_aterms_indices();

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
    cu::DeviceMemory& d_visibilities =
        device.retrieve_device_visibilities(local_id);
    cu::DeviceMemory& d_uvw = device.retrieve_device_uvw(local_id);
    cu::DeviceMemory& d_subgrids = device.retrieve_device_subgrids(local_id);
    cu::DeviceMemory& d_metadata = device.retrieve_device_metadata(local_id);

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

    // Launch splitter kernel
    if (m_use_unified_memory) {
      device.launch_splitter_unified(current_nr_subgrids, grid_size,
                                     subgrid_size, d_metadata, d_subgrids,
                                     grid.data());
    } else {
      cu::DeviceMemory& d_grid = device.retrieve_device_grid();
      device.launch_splitter(current_nr_subgrids, grid_size, subgrid_size,
                             d_metadata, d_subgrids, d_grid);
    }

    // Initialize visibilities to zero
    d_visibilities.zero(executestream);

    // Launch FFT
    device.launch_subgrid_fft(d_subgrids, current_nr_subgrids,
                              ImageDomainToFourierDomain);

    // Launch degridder kernel
    device.launch_degridder(current_time_offset, current_nr_subgrids, grid_size,
                            subgrid_size, image_size, w_step, nr_channels,
                            nr_stations, d_uvw, d_wavenumbers, d_visibilities,
                            d_spheroidal, d_aterms, d_aterms_indices,
                            d_metadata, d_subgrids);
    executestream.record(*gpuFinished[job_id]);

    // Copy visibilities to host
    dtohstream.waitEvent(*gpuFinished[job_id]);
    auto sizeof_visibilities = auxiliary::sizeof_visibilities(
        current_nr_baselines, nr_timesteps, nr_channels);
    dtohstream.memcpyDtoHAsync(visibilities_ptr, d_visibilities,
                               sizeof_visibilities);
    dtohstream.record(*outputCopied[job_id]);

    // Wait for degridder to finish
    gpuFinished[job_id]->synchronize();

    // Report performance
    device.enqueue_report(dtohstream, jobs[job_id].current_nr_timesteps,
                          jobs[job_id].current_nr_subgrids);
  }  // end for bl

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

void Generic::do_degridding(
    const Plan& plan,
    const float w_step,  // in lambda
    const Array1D<float>& shift, const float cell_size,
    const unsigned int kernel_size,  // full width in pixels
    const unsigned int subgrid_size, const Array1D<float>& frequencies,
    Array3D<Visibility<std::complex<float>>>& visibilities,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Grid& grid, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterms_offsets,
    const Array2D<float>& spheroidal) {
#if defined(DEBUG)
  std::cout << "Generic::" << __func__ << std::endl;
#endif

#if defined(DEBUG)
  std::clog << "### Initialize degridding" << std::endl;
#endif
  CUDA::initialize(plan, w_step, shift, cell_size, kernel_size, subgrid_size,
                   frequencies, visibilities, uvw, baselines, aterms,
                   aterms_offsets, spheroidal);

#if defined(DEBUG)
  std::clog << "### Run degridding" << std::endl;
#endif
  run_degridding(plan, w_step, shift, cell_size, kernel_size, subgrid_size,
                 frequencies, visibilities, uvw, baselines, grid, aterms,
                 aterms_offsets, spheroidal);

#if defined(DEBUG)
  std::clog << "### Finish degridding" << std::endl;
#endif
}  // end degridding

void Generic::set_grid(std::shared_ptr<Grid> grid) {
  m_grid = grid;
  InstanceCUDA& device = get_device(0);
  device.set_context();
  device.allocate_device_grid(grid->bytes());
  cu::DeviceMemory& d_grid = device.retrieve_device_grid();
  cu::Stream& htodstream = device.get_htod_stream();
  device.copy_htod(htodstream, d_grid, grid->data(), grid->bytes());
  htodstream.synchronize();
}

std::shared_ptr<Grid> Generic::get_grid() {
  InstanceCUDA& device = get_device(0);
  device.set_context();
  cu::DeviceMemory& d_grid = device.retrieve_device_grid();
  cu::Stream& dtohstream = device.get_dtoh_stream();
  device.copy_dtoh(dtohstream, m_grid->data(), d_grid, m_grid->bytes());
  dtohstream.synchronize();
  return m_grid;
}

}  // namespace cuda
}  // namespace proxy
}  // namespace idg

#include "GenericC.h"
