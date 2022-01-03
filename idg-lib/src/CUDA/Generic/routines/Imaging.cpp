#include "../Generic.h"
#include "InstanceCUDA.h"

using namespace idg::kernel::cuda;
using namespace powersensor;

namespace idg {
namespace proxy {
namespace cuda {

void Generic::run_imaging(
    const Plan& plan, const Array1D<float>& frequencies,
    const Array4D<std::complex<float>>& visibilities,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines, Grid& grid,
    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterms_offsets,
    const Array2D<float>& spheroidal, ImagingMode mode) {
  InstanceCUDA& device = get_device(0);
  const cu::Context& context = device.get_context();

  // Arguments
  auto nr_baselines = visibilities.get_w_dim();
  auto nr_timesteps = visibilities.get_z_dim();
  auto nr_channels = visibilities.get_y_dim();
  auto nr_correlations = visibilities.get_x_dim();
  auto nr_stations = aterms.get_z_dim();
  auto nr_polarizations = grid.get_z_dim();
  auto grid_size = grid.get_x_dim();
  auto cell_size = plan.get_cell_size();
  auto image_size = cell_size * grid_size;
  auto subgrid_size = plan.get_subgrid_size();
  auto w_step = plan.get_w_step();
  auto& shift = plan.get_shift();

  WTileUpdateSet wtile_set;
  if (mode == ImagingMode::mode_gridding) {
    wtile_set = plan.get_wtile_flush_set();
  } else if (mode == ImagingMode::mode_degridding) {
    wtile_set = plan.get_wtile_initialize_set();
  }

  // Convert frequencies to wavenumbers
  Array1D<float> wavenumbers = compute_wavenumbers(frequencies);

  // Aterm indices
  size_t sizeof_aterms_indices =
      auxiliary::sizeof_aterms_indices(nr_baselines, nr_timesteps);
  auto aterms_indices = plan.get_aterm_indices_ptr();

  // Average aterm correction
  size_t sizeof_avg_aterm_correction =
      mode == ImagingMode::mode_gridding && m_avg_aterm_correction.size() > 0
          ? auxiliary::sizeof_avg_aterm_correction(subgrid_size)
          : 0;

  // Configuration
  const unsigned nr_devices = get_num_devices();
  int device_id = 0;  // only one GPU is used

  // Page-locked host memory
  cu::RegisteredMemory h_metadata(context, plan.get_metadata_ptr(),
                                  plan.get_sizeof_metadata());

  // Performance measurements
  m_report->initialize(nr_channels, subgrid_size, grid_size);
  device.set_report(m_report);
  std::vector<State> startStates(nr_devices + 1);
  std::vector<State> endStates(nr_devices + 1);

  // Load streams
  cu::Stream& executestream = device.get_execute_stream();
  cu::Stream& htodstream = device.get_htod_stream();
  cu::Stream& dtohstream = device.get_dtoh_stream();

  // Allocate device memory
  cu::DeviceMemory d_wavenumbers(context, wavenumbers.bytes());
  cu::DeviceMemory d_spheroidal(context, spheroidal.bytes());
  cu::DeviceMemory d_aterms(context, aterms.bytes());
  cu::DeviceMemory d_aterms_indices(context, sizeof_aterms_indices);
  cu::DeviceMemory d_avg_aterm(context, sizeof_avg_aterm_correction);

  // Initialize device memory
  htodstream.memcpyHtoDAsync(d_wavenumbers, wavenumbers.data(),
                             wavenumbers.bytes());
  htodstream.memcpyHtoDAsync(d_spheroidal, spheroidal.data(),
                             spheroidal.bytes());
  htodstream.memcpyHtoDAsync(d_aterms, aterms.data(), aterms.bytes());
  htodstream.memcpyHtoDAsync(d_aterms_indices, aterms_indices,
                             sizeof_aterms_indices);
  htodstream.memcpyHtoDAsync(d_avg_aterm, m_avg_aterm_correction.data(),
                             sizeof_avg_aterm_correction);

  // Plan subgrid fft
  device.plan_subgrid_fft(subgrid_size, nr_polarizations);

  // Get the available device memory, reserving some
  // space for the w-padded tile FFT plan which is allocated
  // in run_wtiles_to_grid and run_wtiles_from_grid.
  size_t bytes_free = device.get_free_memory();
  if (!m_disable_wtiling) {
    bytes_free -= bytes_required_wtiling(plan.get_wtile_initialize_set(),
                                         nr_polarizations, subgrid_size,
                                         image_size, w_step, shift, bytes_free);
  }

  // Initialize jobs
  std::vector<JobData> jobs;
  int jobsize =
      initialize_jobs(nr_baselines, nr_timesteps, nr_channels, subgrid_size,
                      bytes_free, plan, visibilities, uvw, jobs);

  // Allocate device memory for jobs
  std::vector<std::unique_ptr<cu::DeviceMemory>> d_visibilities_;
  std::vector<std::unique_ptr<cu::DeviceMemory>> d_uvw_;
  std::vector<std::unique_ptr<cu::DeviceMemory>> d_subgrids_;
  std::vector<std::unique_ptr<cu::DeviceMemory>> d_metadata_;

  for (int i = 0; i < 2; i++) {
    int max_nr_subgrids = plan.get_max_nr_subgrids(jobsize);
    size_t sizeof_visibilities = auxiliary::sizeof_visibilities(
        jobsize, nr_timesteps, nr_channels, nr_correlations);
    size_t sizeof_uvw = auxiliary::sizeof_uvw(jobsize, nr_timesteps);
    size_t sizeof_subgrids = auxiliary::sizeof_subgrids(
        max_nr_subgrids, subgrid_size, nr_correlations);
    size_t sizeof_metadata = auxiliary::sizeof_metadata(max_nr_subgrids);
    d_visibilities_.emplace_back(
        new cu::DeviceMemory(context, sizeof_visibilities));
    d_uvw_.emplace_back(new cu::DeviceMemory(context, sizeof_uvw));
    d_subgrids_.emplace_back(new cu::DeviceMemory(context, sizeof_subgrids));
    d_metadata_.emplace_back(new cu::DeviceMemory(context, sizeof_metadata));
  }

  // Events
  std::vector<std::unique_ptr<cu::Event>> inputCopied;
  std::vector<std::unique_ptr<cu::Event>> gpuFinished;
  std::vector<std::unique_ptr<cu::Event>> outputCopied;
  for (unsigned bl = 0; bl < nr_baselines; bl += jobsize) {
    inputCopied.emplace_back(new cu::Event(context, CU_EVENT_BLOCKING_SYNC));
    gpuFinished.emplace_back(new cu::Event(context, CU_EVENT_BLOCKING_SYNC));
    outputCopied.emplace_back(new cu::Event(context, CU_EVENT_BLOCKING_SYNC));
  }

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
    auto metadata_ptr = jobs[job_id].metadata_ptr;
    auto uvw_ptr = jobs[job_id].uvw_ptr;
    auto visibilities_ptr = jobs[job_id].visibilities_ptr;

    // Load memory objects
    cu::DeviceMemory& d_visibilities = *d_visibilities_[local_id];
    cu::DeviceMemory& d_uvw = *d_uvw_[local_id];
    cu::DeviceMemory& d_subgrids = *d_subgrids_[local_id];
    cu::DeviceMemory& d_metadata = *d_metadata_[local_id];

    // Copy input data for first job to device
    if (job_id == 0) {
      // Visibilities
      if (mode == ImagingMode::mode_gridding) {
        size_t sizeof_visibilities = auxiliary::sizeof_visibilities(
            current_nr_baselines, nr_timesteps, nr_channels, nr_correlations);
        htodstream.memcpyHtoDAsync(d_visibilities, visibilities_ptr,
                                   sizeof_visibilities);
      }

      // UVW
      size_t sizeof_uvw =
          auxiliary::sizeof_uvw(current_nr_baselines, nr_timesteps);
      htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, sizeof_uvw);

      // Metadata
      size_t sizeof_metadata = auxiliary::sizeof_metadata(current_nr_subgrids);
      htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, sizeof_metadata);

      // Synchronization
      htodstream.record(*inputCopied[job_id]);
    }

    // Copy input data for next job
    if (job_id_next < jobs.size()) {
      unsigned int nr_baselines_next = jobs[job_id_next].current_nr_baselines;

      // Visibilities (gridding only)
      if (mode == ImagingMode::mode_gridding) {
        const void* visibilities_ptr_next = jobs[job_id_next].visibilities_ptr;
        cu::DeviceMemory& d_visibilities_next = *d_visibilities_[local_id_next];
        size_t sizeof_visibilities_next = auxiliary::sizeof_visibilities(
            nr_baselines_next, nr_timesteps, nr_channels, nr_correlations);
        htodstream.memcpyHtoDAsync(d_visibilities_next, visibilities_ptr_next,
                                   sizeof_visibilities_next);
      }

      // UVW
      cu::DeviceMemory& d_uvw_next = *d_uvw_[local_id_next];
      const void* uvw_ptr_next = jobs[job_id_next].uvw_ptr;
      size_t sizeof_uvw_next =
          auxiliary::sizeof_uvw(nr_baselines_next, nr_timesteps);
      htodstream.memcpyHtoDAsync(d_uvw_next, uvw_ptr_next, sizeof_uvw_next);

      // Metadata
      cu::DeviceMemory& d_metadata_next = *d_metadata_[local_id_next];
      const void* metadata_ptr_next = jobs[job_id_next].metadata_ptr;
      unsigned int nr_subgrids_next = jobs[job_id_next].current_nr_subgrids;
      size_t sizeof_metadata_next =
          auxiliary::sizeof_metadata(nr_subgrids_next);
      htodstream.memcpyHtoDAsync(d_metadata_next, metadata_ptr_next,
                                 sizeof_metadata_next);

      // Synchronization
      htodstream.record(*inputCopied[job_id_next]);
    }

    // Wait for output buffer to be free
    if (job_id > 1) {
      if (mode == ImagingMode::mode_gridding) {
        executestream.waitEvent(*gpuFinished[job_id - 2]);
      } else if (mode == ImagingMode::mode_degridding) {
        executestream.waitEvent(*outputCopied[job_id - 2]);
      }
    }

    // Initialize output buffer to zero
    if (mode == ImagingMode::mode_gridding) {
      d_subgrids.zero(executestream);
    } else if (mode == ImagingMode::mode_degridding) {
      d_visibilities.zero(executestream);
    }

    // Wait for input to be copied
    executestream.waitEvent(*inputCopied[job_id]);

    if (mode == ImagingMode::mode_gridding) {
      // Launch gridder kernel
      device.launch_gridder(
          current_time_offset, current_nr_subgrids, nr_polarizations, grid_size,
          subgrid_size, image_size, w_step, nr_channels, nr_stations, shift(0),
          shift(1), d_uvw, d_wavenumbers, d_visibilities, d_spheroidal,
          d_aterms, d_aterms_indices, d_avg_aterm, d_metadata, d_subgrids);

      // Launch FFT
      device.launch_subgrid_fft(d_subgrids, current_nr_subgrids,
                                nr_polarizations, FourierDomainToImageDomain);

      // Launch adder
      if (plan.get_use_wtiles()) {
        auto subgrid_offset = plan.get_subgrid_offset(jobs[job_id].first_bl);
        run_subgrids_to_wtiles(
            nr_polarizations, subgrid_offset, current_nr_subgrids, subgrid_size,
            image_size, w_step, shift, wtile_set, d_subgrids, d_metadata);
      } else {
        if (m_use_unified_memory) {
          device.launch_adder_unified(current_nr_subgrids, grid_size,
                                      subgrid_size, d_metadata, d_subgrids,
                                      get_unified_grid());
        } else {
          device.launch_adder(current_nr_subgrids, nr_polarizations, grid_size,
                              subgrid_size, d_metadata, d_subgrids, *d_grid_);
        }
      }

      executestream.record(*gpuFinished[job_id]);
    } else if (mode == ImagingMode::mode_degridding) {
      // Launch splitter
      if (plan.get_use_wtiles()) {
        auto subgrid_offset = plan.get_subgrid_offset(jobs[job_id].first_bl);
        run_subgrids_from_wtiles(
            nr_polarizations, subgrid_offset, current_nr_subgrids, subgrid_size,
            image_size, w_step, shift, wtile_set, d_subgrids, d_metadata);
      } else {
        if (m_use_unified_memory) {
          device.launch_splitter_unified(current_nr_subgrids, grid_size,
                                         subgrid_size, d_metadata, d_subgrids,
                                         get_unified_grid());
        } else {
          device.launch_splitter(current_nr_subgrids, nr_polarizations,
                                 grid_size, subgrid_size, d_metadata,
                                 d_subgrids, *d_grid_);
        }
      }

      // Launch FFT
      device.launch_subgrid_fft(d_subgrids, current_nr_subgrids,
                                nr_polarizations, ImageDomainToFourierDomain);

      // Launch degridder kernel
      device.launch_degridder(
          current_time_offset, current_nr_subgrids, nr_polarizations, grid_size,
          subgrid_size, image_size, w_step, nr_channels, nr_stations, shift(0),
          shift(1), d_uvw, d_wavenumbers, d_visibilities, d_spheroidal,
          d_aterms, d_aterms_indices, d_metadata, d_subgrids);
      executestream.record(*gpuFinished[job_id]);

      // Copy visibilities to host
      dtohstream.waitEvent(*gpuFinished[job_id]);
      auto sizeof_visibilities = auxiliary::sizeof_visibilities(
          current_nr_baselines, nr_timesteps, nr_channels, nr_correlations);
      dtohstream.memcpyDtoHAsync(visibilities_ptr, d_visibilities,
                                 sizeof_visibilities);
      dtohstream.record(*outputCopied[job_id]);
    }

    // Wait for final compute kernel to finish
    gpuFinished[job_id]->synchronize();

    // Report performance
    device.enqueue_report(executestream, nr_polarizations,
                          jobs[job_id].current_nr_timesteps,
                          jobs[job_id].current_nr_subgrids);
  }  // end for bl

  // Wait for all reports to be printed
  dtohstream.synchronize();

  // End performance measurement
  endStates[device_id] = device.measure();
  endStates[nr_devices] = hostPowerSensor->read();
  m_report->update(Report::device, startStates[device_id],
                   endStates[device_id]);
  m_report->update(Report::host, startStates[nr_devices],
                   endStates[nr_devices]);

  // Update report
  auto total_nr_subgrids = plan.get_nr_subgrids();
  auto total_nr_timesteps = plan.get_nr_timesteps();
  auto total_nr_visibilities = plan.get_nr_visibilities();
  m_report->print_total(nr_correlations, total_nr_timesteps, total_nr_subgrids);
  std::string name = "";
  if (mode == ImagingMode::mode_gridding) {
    name = auxiliary::name_gridding;
  } else if (mode == ImagingMode::mode_degridding) {
    name = auxiliary::name_degridding;
  }
  m_report->print_visibilities(name, total_nr_visibilities);

  // Cleanup
  device.free_subgrid_fft();
}

}  // end namespace cuda
}  // end namespace proxy
}  // end namespace idg