#include "../GenericOptimized.h"
#include "InstanceCUDA.h"

using namespace idg::proxy::cuda;
using namespace idg::proxy::cpu;
using namespace idg::kernel::cpu;
using namespace idg::kernel::cuda;
using namespace powersensor;

namespace idg {
namespace proxy {
namespace hybrid {

void GenericOptimized::do_calibrate_init(
    std::vector<std::unique_ptr<Plan>>&& plans,
    const Array1D<float>& frequencies,
    Array4D<Visibility<std::complex<float>>>&& visibilities,
    Array4D<Visibility<float>>&& weights, Array3D<UVW<float>>&& uvw,
    Array2D<std::pair<unsigned int, unsigned int>>&& baselines,
    const Array2D<float>& spheroidal) {
  InstanceCPU& cpuKernels = cpuProxy->get_kernels();
  cpuKernels.set_report(m_report);

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
  m_report->initialize();
  powersensor::State states[2];
  states[0] = hostPowerSensor->read();

  // Load device
  InstanceCUDA& device = get_device(0);
  device.set_report(m_report);
  const cu::Context& context = device.get_context();

  // Load stream
  cu::Stream& htodstream = device.get_htod_stream();
  cu::Stream& dtohstream = device.get_dtoh_stream();
  cu::Stream& executestream = device.get_execute_stream();

  // Find max number of subgrids
  unsigned int max_nr_subgrids = 0;

  // Free buffers and clear the gridding state
  free_buffers();
  m_gridding_state.nr_stations = 0;

  m_buffers.d_aterms.reset(new cu::DeviceMemory(context, 0));

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
    void* visibilities_ptr = visibilities.data(antenna_nr);
    void* weights_ptr = weights.data(antenna_nr);
    void* uvw_ptr = uvw.data(antenna_nr);

    // Allocate and initialize device memory for current antenna
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
    m_buffers.d_metadata_.emplace_back(
        new cu::DeviceMemory(context, sizeof_metadata));
    m_buffers.d_subgrids_.emplace_back(
        new cu::DeviceMemory(context, sizeof_subgrids));
    m_buffers.d_visibilities_.emplace_back(
        new cu::DeviceMemory(context, sizeof_visibilities));
    m_buffers.d_weights_.emplace_back(
        new cu::DeviceMemory(context, sizeof_weights));
    m_buffers.d_uvw_.emplace_back(new cu::DeviceMemory(context, sizeof_uvw));
    m_buffers.d_aterms_indices_.emplace_back(
        new cu::DeviceMemory(context, sizeof_aterm_idx));
    cu::DeviceMemory& d_metadata = *m_buffers.d_metadata_[antenna_nr];
    cu::DeviceMemory& d_subgrids = *m_buffers.d_subgrids_[antenna_nr];
    cu::DeviceMemory& d_visibilities = *m_buffers.d_visibilities_[antenna_nr];
    cu::DeviceMemory& d_weights = *m_buffers.d_weights_[antenna_nr];
    cu::DeviceMemory& d_uvw = *m_buffers.d_uvw_[antenna_nr];
    cu::DeviceMemory& d_aterm_idx = *m_buffers.d_aterms_indices_[antenna_nr];

    // Copy metadata to device
    htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, sizeof_metadata);

    // Splitter kernel
    if (w_step == 0.0) {
      cpuKernels.run_splitter(nr_subgrids, grid_size, subgrid_size,
                              metadata_ptr, subgrids_ptr, grid_ptr);
    } else if (plans[antenna_nr]->get_use_wtiles()) {
      WTileUpdateSet wtile_initialize_set =
          plans[antenna_nr]->get_wtile_initialize_set();
      if (!m_disable_wtiling_gpu) {
        // Initialize subgrid FFT
        device.plan_subgrid_fft(subgrid_size, nr_subgrids);

        // Wait for metadata to be copied
        htodstream.synchronize();

        // Create subgrids
        run_subgrids_from_wtiles(0 /* subgrid_offset */, nr_subgrids,
                                 subgrid_size, image_size, w_step, shift,
                                 wtile_initialize_set, d_subgrids, d_metadata);
        executestream.synchronize();

        // Copy subgrids to host
        dtohstream.memcpyDtoHAsync(subgrids_ptr, d_subgrids, sizeof_subgrids);
        dtohstream.synchronize();
      } else {
        cpuKernels.run_splitter_wtiles(
            nr_subgrids, grid_size, subgrid_size, image_size, w_step,
            shift.data(), 0 /* subgrid_offset */, wtile_initialize_set,
            metadata_ptr, subgrids_ptr, grid_ptr);
      }
    } else {
      cpuKernels.run_splitter_wstack(nr_subgrids, grid_size, subgrid_size,
                                     metadata_ptr, subgrids_ptr, grid_ptr);
    }

    // Copy data to device
    htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, sizeof_metadata);
    htodstream.memcpyHtoDAsync(d_visibilities, visibilities_ptr,
                               sizeof_visibilities);
    htodstream.memcpyHtoDAsync(d_weights, weights_ptr, sizeof_weights);
    htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, sizeof_uvw);
    htodstream.memcpyHtoDAsync(d_aterm_idx, aterm_idx_ptr, sizeof_aterm_idx);

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

    // Copy subgrids to device
    htodstream.memcpyHtoDAsync(d_subgrids, subgrids_ptr, sizeof_subgrids);
    htodstream.synchronize();
  }  // end for antennas

  // End performance measurement
  states[1] = hostPowerSensor->read();
  m_report->update(Report::host, states[0], states[1]);
  m_report->print_total(0, 0);

  // Set calibration state member variables
  m_calibrate_state.plans = std::move(plans);
  m_calibrate_state.nr_baselines = nr_baselines;
  m_calibrate_state.nr_timesteps = nr_timesteps;
  m_calibrate_state.nr_channels = nr_channels;

  // Initialize wavenumbers
  m_buffers.d_wavenumbers.reset(
      new cu::DeviceMemory(context, wavenumbers.bytes()));
  cu::DeviceMemory& d_wavenumbers = *m_buffers.d_wavenumbers;
  htodstream.memcpyHtoDAsync(d_wavenumbers, wavenumbers.data(),
                             wavenumbers.bytes());

  // Allocate device memory for l,m,n and phase offset
  auto sizeof_lmnp =
      max_nr_subgrids * subgrid_size * subgrid_size * 4 * sizeof(float);
  m_buffers.d_lmnp.reset(new cu::DeviceMemory(context, sizeof_lmnp));

  // Allocate memory for sums (horizontal and vertical)
  auto total_nr_timesteps = nr_baselines * nr_timesteps;
  auto sizeof_sums = max_nr_terms * nr_correlations * total_nr_timesteps *
                     nr_channels * sizeof(std::complex<float>);
  for (unsigned int i = 0; i < 2; i++) {
    m_buffers.d_sums_.emplace_back(new cu::DeviceMemory(context, sizeof_sums));
  }
}

void GenericOptimized::do_calibrate_update(
    const int antenna_nr, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array4D<Matrix2x2<std::complex<float>>>& aterm_derivatives,
    Array3D<double>& hessian, Array2D<double>& gradient, double& residual) {
  // Check if the proxy is still in calibrate state
  // A calibrate_init call brings the proxy in calibrate state
  // A (de)gridding call brings the proxy in gridding state
  // If the proxy is gridding state here, an exception is thrown
  if (m_gridding_state.nr_stations) {
    throw std::runtime_error(
        "calibrate_update() was called while the proxy is in gridding state");
  }
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
    m_report->initialize(nr_channels, subgrid_size, 0, nr_terms);
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
  m_report->update_total(nr_subgrids, nr_timesteps, nr_visibilities);
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
  m_report->print_total(total_nr_timesteps, total_nr_subgrids);
  m_report->print_visibilities(auxiliary::name_calibrate);
}

}  // namespace hybrid
}  // namespace proxy
}  // namespace idg
