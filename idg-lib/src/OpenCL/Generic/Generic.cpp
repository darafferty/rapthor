// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <clFFT.h>

#include "Generic.h"

#include "InstanceOpenCL.h"
#include "PowerRecord.h"

/*
    Toggle planning and execution of Fourier transformations on and off
        The clFFT library contains memory leaks, which makes it much harder
        to find and resolve issues in non-library code. This option disables
        usage of the library so that they can be resolved
*/
#define ENABLE_FFT 1

using namespace std;
using namespace idg::kernel::opencl;
using namespace powersensor;

namespace idg {
namespace proxy {
namespace opencl {

// Constructor
Generic::Generic() {
#if defined(DEBUG)
  cout << "Generic::" << __func__ << endl;
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
  cout << __func__ << endl;
#endif

  // Constants
  auto grid_size = grid.get_x_dim();

  // Load device
  InstanceOpenCL& device = get_device(0);

  // Command queue
  cl::CommandQueue& queue = device.get_execute_queue();

  // Power measurements
  report.initialize(0, 0, grid_size);
  device.set_report(report);
  PowerRecord powerRecords[4];
  State powerStates[4];
  powerStates[0] = hostPowerSensor->read();
  powerStates[2] = device.measure();

  // Device memory
  cl::Buffer& d_grid = device.get_device_grid(grid_size);

  // Perform fft shift
  device.shift(grid);

  // Copy grid to device
  device.measure(powerRecords[0], queue);
  writeBuffer(queue, d_grid, CL_FALSE, grid.data());
  device.measure(powerRecords[1], queue);

// Create FFT plan
#if ENABLE_FFT
  device.plan_fft(grid_size, 1);
#endif

  // Launch FFT
#if ENABLE_FFT
  device.launch_fft(d_grid, direction);
#endif

  // Copy grid to host
  device.measure(powerRecords[2], queue);
  readBuffer(queue, d_grid, CL_FALSE, grid.data());
  device.measure(powerRecords[3], queue);
  queue.finish();

  // Perform fft shift
  device.shift(grid);

  // Perform fft scaling
  complex<float> scale = complex<float>(2, 0);
  if (direction == FourierDomainToImageDomain) {
    device.scale(grid, scale);
  }

  // End measurements
  powerStates[1] = hostPowerSensor->read();
  powerStates[3] = device.measure();

  // Report performance
  report.update_input(powerRecords[0].state, powerRecords[1].state);
  report.update_output(powerRecords[2].state, powerRecords[3].state);
  report.update_host(powerStates[0], powerStates[1]);
  report.print_total();
  report.print_device(powerRecords[0].state, powerRecords[3].state);
}  // end transform

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
  cout << __func__ << endl;
#endif

  Array1D<float> wavenumbers = compute_wavenumbers(frequencies);

  // Checks arguments
  if (kernel_size <= 0 || kernel_size >= subgrid_size - 1) {
    throw invalid_argument("0 < kernel_size < subgrid_size-1 not true");
  }

  // Arguments
  auto nr_baselines = visibilities.get_z_dim();
  auto nr_timesteps = visibilities.get_y_dim();
  auto nr_channels = visibilities.get_x_dim();
  auto nr_stations = aterms.get_z_dim();
  auto nr_timeslots = aterms.get_w_dim();
  auto nr_correlations = grid.get_z_dim();
  auto grid_size = grid.get_x_dim();
  auto image_size = cell_size * grid_size;

  // Configuration
  const int nr_devices = get_num_devices();
  const int nr_streams = 2;

  // Initialize metadata
  std::vector<int> jobsize_ = compute_jobsize(plan, nr_timesteps, nr_channels,
                                              subgrid_size, nr_streams);

  // Initialize host memory
  cl::Context& context = get_context();
  InstanceOpenCL& device = get_device(0);
  cl::Buffer h_visibilities = device.get_host_visibilities(
      nr_baselines, nr_timesteps, nr_channels, visibilities.data());
  cl::Buffer h_uvw =
      device.get_host_uvw(nr_baselines, nr_timesteps, uvw.data());

  // Initialize device memory
  for (int d = 0; d < nr_devices; d++) {
    InstanceOpenCL& device = get_device(d);
    device.set_report(report);
    cl::CommandQueue& htodqueue = device.get_htod_queue();
    cl::Buffer& d_wavenumbers = device.get_device_wavenumbers(nr_channels);
    cl::Buffer& d_spheroidal = device.get_device_spheroidal(subgrid_size);
    cl::Buffer& d_aterms =
        device.get_device_aterms(nr_stations, nr_timeslots, subgrid_size);
    cl::Buffer& d_grid = device.get_device_grid(grid_size);
    writeBufferBatched(htodqueue, d_wavenumbers, CL_FALSE, wavenumbers.data());
    writeBufferBatched(htodqueue, d_spheroidal, CL_FALSE, spheroidal.data());
    writeBufferBatched(htodqueue, d_aterms, CL_FALSE, aterms.data());
    writeBufferBatched(htodqueue, d_grid, CL_FALSE, grid.data());
  }

  // Performance measurements
  report.initialize(nr_channels, subgrid_size, grid_size);
  vector<State> startStates(nr_devices + 1);
  vector<State> endStates(nr_devices + 1);

#pragma omp parallel num_threads(nr_devices* nr_streams)
  {
    int global_id = omp_get_thread_num();
    int device_id = global_id / nr_streams;
    int local_id = global_id % nr_streams;
    int jobsize = jobsize_[device_id];
    int max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);

    // Limit jobsize
    jobsize = min(jobsize, nr_baselines);

    // Load device
    InstanceOpenCL& device = get_device(device_id);

    // Load OpenCL objects
    cl::CommandQueue& executequeue = device.get_execute_queue();
    cl::CommandQueue& htodqueue = device.get_htod_queue();

    // Load memory objects
    cl::Buffer& d_grid = device.get_device_grid();
    cl::Buffer& d_wavenumbers = device.get_device_wavenumbers();
    cl::Buffer& d_spheroidal = device.get_device_spheroidal();
    cl::Buffer& d_aterms = device.get_device_aterms();

    // Events
    vector<cl::Event> inputReady(1);
    vector<cl::Event> outputReady(1);
    htodqueue.enqueueMarkerWithWaitList(NULL, &outputReady[0]);

    // Allocate private device memory
    auto sizeof_visibilities =
        auxiliary::sizeof_visibilities(jobsize, nr_timesteps, nr_channels);
    auto sizeof_uvw = auxiliary::sizeof_uvw(jobsize, nr_timesteps);
    auto sizeof_subgrids =
        auxiliary::sizeof_subgrids(max_nr_subgrids, subgrid_size);
    auto sizeof_metadata = auxiliary::sizeof_metadata(max_nr_subgrids);
    cl::Buffer d_visibilities =
        cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_visibilities);
    cl::Buffer d_uvw = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_uvw);
    cl::Buffer d_subgrids =
        cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_subgrids);
    cl::Buffer d_metadata =
        cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_metadata);

// Create FFT plan
#if ENABLE_FFT
    if (local_id == 0) {
      device.plan_fft(subgrid_size, max_nr_subgrids);
    }
#endif

    // Performance measurement
    PowerRecord powerRecords[4];
    if (local_id == 0) {
      startStates[device_id] = device.measure();
    }
    if (global_id == 0) {
      startStates[nr_devices] = hostPowerSensor->read();
    }

#pragma omp barrier
#pragma omp for schedule(dynamic)
    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
      unsigned int first_bl, last_bl, current_nr_baselines;
      plan.initialize_job(nr_baselines, jobsize, bl, &first_bl, &last_bl,
                          &current_nr_baselines);
      if (current_nr_baselines == 0) continue;

      // Initialize iteration
      auto current_nr_subgrids =
          plan.get_nr_subgrids(first_bl, current_nr_baselines);
      auto current_nr_timesteps =
          plan.get_nr_timesteps(first_bl, current_nr_baselines);
      auto uvw_offset = first_bl * auxiliary::sizeof_uvw(1, nr_timesteps);
      auto visibilities_offset = first_bl * auxiliary::sizeof_visibilities(
                                                1, nr_timesteps, nr_channels);

#pragma omp critical(lock)
      {
        // Copy input data to device
        // htodqueue.enqueueBarrierWithWaitList(&outputReady, NULL);
        htodqueue.enqueueCopyBuffer(
            h_visibilities, d_visibilities, visibilities_offset, 0,
            auxiliary::sizeof_visibilities(current_nr_baselines, nr_timesteps,
                                           nr_channels));
        htodqueue.enqueueCopyBuffer(
            h_uvw, d_uvw, uvw_offset, 0,
            auxiliary::sizeof_uvw(current_nr_baselines, nr_timesteps));
        htodqueue.enqueueWriteBuffer(
            d_metadata, CL_FALSE, 0,
            auxiliary::sizeof_metadata(current_nr_subgrids),
            plan.get_metadata_ptr(first_bl));
        htodqueue.enqueueMarkerWithWaitList(NULL, &inputReady[0]);

        // Launch gridder kernel
        executequeue.enqueueBarrierWithWaitList(&inputReady, NULL);
        device.launch_gridder(
            current_nr_timesteps, current_nr_subgrids, grid_size, subgrid_size,
            image_size, w_step, nr_channels, nr_stations, d_uvw, d_wavenumbers,
            d_visibilities, d_spheroidal, d_aterms, d_metadata, d_subgrids);

        // Launch FFT
#if ENABLE_FFT
        device.launch_fft(d_subgrids, FourierDomainToImageDomain);
#endif

        // Launch adder kernel
        device.launch_adder(current_nr_subgrids, grid_size, subgrid_size,
                            d_metadata, d_subgrids, d_grid);
        device.enqueue_report(executequeue, current_nr_timesteps,
                              current_nr_subgrids);
        executequeue.enqueueMarkerWithWaitList(NULL, &outputReady[0]);
      }

      outputReady[0].wait();
    }  // end for bl

    // Wait for all jobs to finish
    executequeue.finish();

    // End power measurement
    if (local_id == 0) {
      endStates[device_id] = device.measure();
    }
    if (global_id == 0) {
      endStates[nr_devices] = hostPowerSensor->read();
      report.update_host(startStates[nr_devices], endStates[nr_devices]);
    }
  }  // end omp parallel

  // Add grids
  for (int d = 0; d < nr_devices; d++) {
    InstanceOpenCL& device = get_device(d);
    cl::CommandQueue queue = device.get_dtoh_queue();
    cl::Buffer& d_grid = device.get_device_grid(grid_size);
    float2* grid_dst = (float2*)grid.data();
    float2* grid_src = (float2*)mapBuffer(queue, d_grid, CL_TRUE, CL_MAP_READ);

#pragma omp parallel for
    for (unsigned i = 0; i < grid_size * grid_size * nr_correlations; i++) {
      grid_dst[i] += grid_src[i];
    }

    unmapBuffer(queue, d_grid, grid_src);
  }

  // Report performance
  auto total_nr_subgrids = plan.get_nr_subgrids();
  auto total_nr_timesteps = plan.get_nr_timesteps();
  auto total_nr_visibilities = plan.get_nr_visibilities();
  report.print_total(total_nr_timesteps, total_nr_subgrids);
  startStates.pop_back();
  endStates.pop_back();
  report.print_devices(startStates, endStates);
  report.print_visibilities(auxiliary::name_gridding, total_nr_visibilities);
}  // end gridding

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
  cout << __func__ << endl;
#endif

  Array1D<float> wavenumbers = compute_wavenumbers(frequencies);

  // Checks arguments
  if (kernel_size <= 0 || kernel_size >= subgrid_size - 1) {
    throw invalid_argument("0 < kernel_size < subgrid_size-1 not true");
  }

  // Arguments
  auto nr_baselines = visibilities.get_z_dim();
  auto nr_timesteps = visibilities.get_y_dim();
  auto nr_channels = visibilities.get_x_dim();
  auto nr_stations = aterms.get_z_dim();
  auto nr_timeslots = aterms.get_w_dim();
  auto grid_size = grid.get_x_dim();
  auto image_size = cell_size * grid_size;

  // Configuration
  const int nr_devices = get_num_devices();
  const int nr_streams = 2;

  // Initialize metadata
  std::vector<int> jobsize_ = compute_jobsize(plan, nr_timesteps, nr_channels,
                                              subgrid_size, nr_streams);

  // Initialize host memory
  cl::Context& context = get_context();
  InstanceOpenCL& device = get_device(0);
  cl::CommandQueue& htodqueue = device.get_htod_queue();
  cl::Buffer h_visibilities = device.get_host_visibilities(
      nr_baselines, nr_timesteps, nr_channels, visibilities.data());
  cl::Buffer h_uvw =
      device.get_host_uvw(nr_baselines, nr_timesteps, uvw.data());

  // Initialize device memory
  for (int d = 0; d < nr_devices; d++) {
    InstanceOpenCL& device = get_device(d);
    device.set_report(report);
    cl::CommandQueue& htodqueue = device.get_htod_queue();
    cl::Buffer& d_wavenumbers = device.get_device_wavenumbers(nr_channels);
    cl::Buffer& d_spheroidal = device.get_device_spheroidal(subgrid_size);
    cl::Buffer& d_aterms =
        device.get_device_aterms(nr_stations, nr_timeslots, subgrid_size);
    cl::Buffer& d_grid = device.get_device_grid(grid_size);
    writeBufferBatched(htodqueue, d_wavenumbers, CL_FALSE, wavenumbers.data());
    writeBufferBatched(htodqueue, d_spheroidal, CL_FALSE, spheroidal.data());
    writeBufferBatched(htodqueue, d_aterms, CL_FALSE, aterms.data());
    writeBufferBatched(htodqueue, d_grid, CL_FALSE, grid.data());
  }

  // Performance measurements
  report.initialize(nr_channels, subgrid_size, grid_size);
  vector<State> startStates(nr_devices + 1);
  vector<State> endStates(nr_devices + 1);

#pragma omp parallel num_threads(nr_devices* nr_streams)
  {
    int global_id = omp_get_thread_num();
    int device_id = global_id / nr_streams;
    int local_id = global_id % nr_streams;
    int jobsize = jobsize_[device_id];
    int max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);

    // Limit jobsize
    jobsize = min(jobsize, nr_baselines);

    // Load device
    InstanceOpenCL& device = get_device(device_id);

    // Load OpenCL objects
    cl::CommandQueue& executequeue = device.get_execute_queue();
    cl::CommandQueue& dtohqueue = device.get_dtoh_queue();

    // Load memory objects
    cl::Buffer& d_grid = device.get_device_grid();
    cl::Buffer& d_wavenumbers = device.get_device_wavenumbers();
    cl::Buffer& d_spheroidal = device.get_device_spheroidal();
    cl::Buffer& d_aterms = device.get_device_aterms();

    // Events
    vector<cl::Event> inputReady(1);
    vector<cl::Event> outputReady(1);
    vector<cl::Event> outputFree(1);
    htodqueue.enqueueMarkerWithWaitList(NULL, &outputFree[0]);

    // Allocate private device memory
    auto sizeof_visibilities =
        auxiliary::sizeof_visibilities(jobsize, nr_timesteps, nr_channels);
    auto sizeof_uvw = auxiliary::sizeof_uvw(jobsize, nr_timesteps);
    auto sizeof_subgrids =
        auxiliary::sizeof_subgrids(max_nr_subgrids, subgrid_size);
    auto sizeof_metadata = auxiliary::sizeof_metadata(max_nr_subgrids);
    cl::Buffer d_visibilities =
        cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_visibilities);
    cl::Buffer d_uvw = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_uvw);
    cl::Buffer d_subgrids =
        cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_subgrids);
    cl::Buffer d_metadata =
        cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_metadata);

// Create FFT plan
#if ENABLE_FFT
    if (local_id == 0) {
      device.plan_fft(subgrid_size, max_nr_subgrids);
    }
#endif

    // Performance measurement
    if (local_id == 0) {
      startStates[device_id] = device.measure();
    }
    if (global_id == 0) {
      startStates[nr_devices] = hostPowerSensor->read();
    }

#pragma omp barrier
#pragma omp for schedule(dynamic)
    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
      unsigned int first_bl, last_bl, current_nr_baselines;
      plan.initialize_job(nr_baselines, jobsize, bl, &first_bl, &last_bl,
                          &current_nr_baselines);
      if (current_nr_baselines == 0) continue;

      // Initialize iteration
      auto current_nr_subgrids =
          plan.get_nr_subgrids(first_bl, current_nr_baselines);
      auto current_nr_timesteps =
          plan.get_nr_timesteps(first_bl, current_nr_baselines);
      auto uvw_offset = first_bl * auxiliary::sizeof_uvw(1, nr_timesteps);
      auto visibilities_offset = first_bl * auxiliary::sizeof_visibilities(
                                                1, nr_timesteps, nr_channels);

#pragma omp critical(lock)
      {
        // Initialize visibilities to zero
        zeroBuffer(htodqueue, d_visibilities);

        // Copy input data to device
        htodqueue.enqueueBarrierWithWaitList(&outputFree, NULL);
        htodqueue.enqueueCopyBuffer(
            h_uvw, d_uvw, uvw_offset, 0,
            auxiliary::sizeof_uvw(current_nr_baselines, nr_timesteps));
        htodqueue.enqueueWriteBuffer(
            d_metadata, CL_FALSE, 0,
            auxiliary::sizeof_metadata(current_nr_subgrids),
            plan.get_metadata_ptr(first_bl));
        htodqueue.enqueueMarkerWithWaitList(NULL, &inputReady[0]);

        // Launch splitter kernel
        executequeue.enqueueMarkerWithWaitList(&inputReady, NULL);
        device.launch_splitter(current_nr_subgrids, grid_size, subgrid_size,
                               d_metadata, d_subgrids, d_grid);

// Launch FFT
#if ENABLE_FFT
        device.launch_fft(d_subgrids, ImageDomainToFourierDomain);
#endif

        // Launch degridder kernel
        device.launch_degridder(
            current_nr_timesteps, current_nr_subgrids, grid_size, subgrid_size,
            image_size, w_step, nr_channels, nr_stations, d_uvw, d_wavenumbers,
            d_visibilities, d_spheroidal, d_aterms, d_metadata, d_subgrids);
        executequeue.enqueueMarkerWithWaitList(NULL, &outputReady[0]);

        // Copy visibilities to host
        dtohqueue.enqueueBarrierWithWaitList(&outputReady, NULL);
        dtohqueue.enqueueCopyBuffer(
            d_visibilities, h_visibilities, 0, visibilities_offset,
            auxiliary::sizeof_visibilities(current_nr_baselines, nr_timesteps,
                                           nr_channels));
        device.enqueue_report(executequeue, current_nr_timesteps,
                              current_nr_subgrids);
        dtohqueue.enqueueMarkerWithWaitList(NULL, &outputFree[0]);
      }

      outputFree[0].wait();
    }  // end for bl

    // Wait for all jobs to finish
    dtohqueue.finish();

    // End measurement
    if (local_id == 0) {
      endStates[device_id] = device.measure();
    }
    if (global_id == 0) {
      endStates[nr_devices] = hostPowerSensor->read();
      report.update_host(startStates[nr_devices], endStates[nr_devices]);
    }
  }  // end omp parallel

  // Report performance
  auto total_nr_subgrids = plan.get_nr_subgrids();
  auto total_nr_timesteps = plan.get_nr_timesteps();
  auto total_nr_visibilities = plan.get_nr_visibilities();
  report.print_total(total_nr_timesteps, total_nr_subgrids);
  startStates.pop_back();
  endStates.pop_back();
  report.print_devices(startStates, endStates);
  report.print_visibilities(auxiliary::name_degridding, total_nr_visibilities);
}  // end degridding

}  // namespace opencl
}  // namespace proxy
}  // namespace idg

#include "GenericC.h"
