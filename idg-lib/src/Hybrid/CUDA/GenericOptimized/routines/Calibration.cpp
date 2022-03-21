#include <algorithm>

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
    std::vector<std::vector<std::unique_ptr<Plan>>>&& plans,
    const Array2D<float>& frequencies,
    Array6D<std::complex<float>>&& visibilities, Array6D<float>&& weights,
    Array3D<UVW<float>>&& uvw,
    Array2D<std::pair<unsigned int, unsigned int>>&& baselines,
    const Array2D<float>& spheroidal) {
  std::cout << "GenericOptimized::" << __func__ << std::endl;

  auto cpuKernels = cpuProxy->get_kernels();
  cpuKernels->set_report(m_report);

  // Arguments
  auto nr_antennas = plans.size();
  auto nr_polarizations = m_grid->get_z_dim();
  auto grid_size = m_grid->get_x_dim();
  auto image_size = m_cache_state.cell_size * grid_size;
  auto w_step = m_cache_state.w_step;
  auto subgrid_size = m_cache_state.subgrid_size;
  auto nr_channel_blocks = visibilities.get_e_dim();
  auto nr_baselines = visibilities.get_d_dim();
  auto nr_timesteps = visibilities.get_c_dim();
  auto nr_channels_per_block = visibilities.get_b_dim();
  auto nr_correlations = visibilities.get_a_dim();
  auto max_nr_terms = m_calibrate_max_nr_terms;
  auto& shift = m_cache_state.shift;

  std::vector<Array1D<float>> wavenumbers;
  wavenumbers.reserve(nr_channel_blocks);
  for (unsigned int channel_block = 0; channel_block < nr_channel_blocks;
       channel_block++) {
    Array1D<float> frequencies_channel_block(frequencies.data(channel_block),
                                             nr_channels_per_block);
    wavenumbers.push_back(compute_wavenumbers(frequencies_channel_block));
  }

  // Allocate subgrids for all antennas and channel_blocks
  std::vector<std::vector<Array4D<std::complex<float>>>> subgrids(nr_antennas);

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

  m_calibrate_state.d_metadata.clear();
  m_calibrate_state.d_subgrids.clear();
  m_calibrate_state.d_visibilities.clear();
  m_calibrate_state.d_weights.clear();
  m_calibrate_state.d_uvw.clear();
  m_calibrate_state.d_aterms_indices.clear();

  // Find max number of subgrids
  unsigned int max_nr_subgrids = 0;
  for (unsigned int antenna_nr = 0; antenna_nr < nr_antennas; antenna_nr++) {
    for (unsigned int channel_block = 0; channel_block < nr_channel_blocks;
         channel_block++) {
      unsigned int nr_subgrids =
          plans[antenna_nr][channel_block]->get_nr_subgrids();
      if (nr_subgrids > max_nr_subgrids) {
        max_nr_subgrids = nr_subgrids;
      }
    }
  }

  // Allocate device memory

  auto sizeof_metadata = auxiliary::sizeof_metadata(max_nr_subgrids);
  auto sizeof_subgrids = auxiliary::sizeof_subgrids(
      max_nr_subgrids, subgrid_size, nr_polarizations);
  auto sizeof_visibilities = auxiliary::sizeof_visibilities(
      nr_baselines, nr_timesteps, nr_channels_per_block, nr_correlations);
  auto sizeof_weights = auxiliary::sizeof_weights(
      nr_baselines, nr_timesteps, nr_channels_per_block, nr_correlations);
  auto sizeof_uvw = auxiliary::sizeof_uvw(nr_baselines, nr_timesteps);
  auto sizeof_aterm_idx =
      auxiliary::sizeof_aterms_indices(nr_baselines, nr_timesteps);
  auto sizeof_wavenumbers =
      auxiliary::sizeof_wavenumbers(nr_channels_per_block);

  m_calibrate_state.d_metadata.emplace_back(
      new cu::DeviceMemory(context, sizeof_metadata));
  m_calibrate_state.d_subgrids.emplace_back(
      new cu::DeviceMemory(context, sizeof_subgrids));
  m_calibrate_state.d_visibilities.emplace_back(
      new cu::DeviceMemory(context, sizeof_visibilities));
  m_calibrate_state.d_weights.emplace_back(
      new cu::DeviceMemory(context, sizeof_weights));
  m_calibrate_state.d_uvw.emplace_back(
      new cu::DeviceMemory(context, sizeof_uvw));
  m_calibrate_state.d_aterms_indices.emplace_back(
      new cu::DeviceMemory(context, sizeof_aterm_idx));

  const unsigned int buffer_nr = 0;
  cu::DeviceMemory& d_metadata = *m_calibrate_state.d_metadata[buffer_nr];
  cu::DeviceMemory& d_subgrids = *m_calibrate_state.d_subgrids[buffer_nr];

  // Create subgrids for every antenna and channel_block
  for (unsigned int antenna_nr = 0; antenna_nr < nr_antennas; antenna_nr++) {
    subgrids[antenna_nr].reserve(nr_channel_blocks);
    for (unsigned int channel_block = 0; channel_block < nr_channel_blocks;
         channel_block++) {
      // Allocate subgrids for current antenna
      unsigned int nr_subgrids =
          plans[antenna_nr][channel_block]->get_nr_subgrids();
      Array4D<std::complex<float>> subgrids1(nr_subgrids, nr_correlations,
                                             subgrid_size, subgrid_size);
      auto sizeof_subgrids1 = auxiliary::sizeof_subgrids(
          nr_subgrids, subgrid_size, nr_polarizations);
      auto sizeof_metadata1 = auxiliary::sizeof_metadata(nr_subgrids);

      // Get data pointers
      auto metadata_ptr = plans[antenna_nr][channel_block]->get_metadata_ptr();
      auto subgrids_ptr = subgrids1.data();
      auto grid_ptr = m_grid->data();

      // Copy metadata to device
      htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, sizeof_metadata1);

      // Splitter kernel
      if (w_step == 0.0) {
        cpuKernels->run_splitter(nr_subgrids, nr_polarizations, grid_size,
                                 subgrid_size, metadata_ptr, subgrids_ptr,
                                 grid_ptr);
      } else if (plans[antenna_nr][channel_block]->get_use_wtiles()) {
        WTileUpdateSet wtile_initialize_set =
            plans[antenna_nr][channel_block]->get_wtile_initialize_set();
        if (!m_disable_wtiling_gpu) {
          // Initialize subgrid FFT
          // device.plan_subgrid_fft(subgrid_size, nr_polarizations);

          // Wait for metadata to be copied
          htodstream.synchronize();

          // Create subgrids
          const unsigned int subgrid_offset = 0;
          run_subgrids_from_wtiles(nr_polarizations, subgrid_offset,
                                   nr_subgrids, subgrid_size, image_size,
                                   w_step, shift, wtile_initialize_set,
                                   d_subgrids, d_metadata);
          executestream.synchronize();

          // Copy subgrids to host
          dtohstream.memcpyDtoHAsync(subgrids_ptr, d_subgrids,
                                     sizeof_subgrids1);
          dtohstream.synchronize();
        } else {
          cpuKernels->run_splitter_wtiles(
              nr_subgrids, nr_polarizations, grid_size, subgrid_size,
              image_size, w_step, shift.data(), 0 /* subgrid_offset */,
              wtile_initialize_set, metadata_ptr, subgrids_ptr, grid_ptr);
        }
      } else {
        cpuKernels->run_splitter_wstack(nr_subgrids, nr_polarizations,
                                        grid_size, subgrid_size, metadata_ptr,
                                        subgrids_ptr, grid_ptr);
      }

      // FFT kernel
      cpuKernels->run_subgrid_fft(grid_size, subgrid_size,
                                  nr_subgrids * nr_polarizations, subgrids_ptr,
                                  CUFFT_FORWARD);

      // Apply spheroidal
      for (int i = 0; i < (int)nr_subgrids; i++) {
        for (unsigned int pol = 0; pol < nr_correlations; pol++) {
          for (int j = 0; j < subgrid_size; j++) {
            for (int k = 0; k < subgrid_size; k++) {
              // Apply FFT shift (swapping corner and centre) to subgrid indices
              const int y = (j + (subgrid_size / 2)) % subgrid_size;
              const int x = (k + (subgrid_size / 2)) % subgrid_size;
              subgrids1(i, pol, y, x) *= spheroidal(j, k);
            }
          }
        }
      }

      subgrids[antenna_nr].push_back(std::move(subgrids1));

    }  // end for channel_blocks
  }    // end for antennas

  // End performance measurement
  states[1] = hostPowerSensor->read();
  m_report->update(Report::host, states[0], states[1]);
  m_report->print_total(0, 0, 0);

  // Set calibration state member variables
  m_calibrate_state.wavenumbers = std::move(wavenumbers);
  m_calibrate_state.plans = std::move(plans);
  m_calibrate_state.uvw = std::move(uvw);
  m_calibrate_state.visibilities = std::move(visibilities);
  m_calibrate_state.weights = std::move(weights);
  m_calibrate_state.subgrids = std::move(subgrids);
  m_calibrate_state.nr_baselines = nr_baselines;
  m_calibrate_state.nr_timesteps = nr_timesteps;
  m_calibrate_state.nr_channels_per_block = nr_channels_per_block;
  m_calibrate_state.nr_channel_blocks = nr_channel_blocks;

  // Initialize wavenumbers
  m_calibrate_state.d_wavenumbers.reset(
      new cu::DeviceMemory(context, sizeof_wavenumbers));

  // Allocate device memory for l,m,n and phase offset
  auto sizeof_lmnp =
      max_nr_subgrids * subgrid_size * subgrid_size * 4 * sizeof(float);
  m_calibrate_state.d_lmnp.reset(new cu::DeviceMemory(context, sizeof_lmnp));

  // Allocate memory for sums (horizontal and vertical)
  auto total_nr_timesteps = nr_baselines * nr_timesteps;
  auto sizeof_sums = max_nr_terms * nr_correlations * total_nr_timesteps *
                     nr_channels_per_block * sizeof(std::complex<float>);
  m_calibrate_state.d_sums_x.reset(new cu::DeviceMemory(context, sizeof_sums));
  m_calibrate_state.d_sums_y.reset(new cu::DeviceMemory(context, sizeof_sums));
}

void GenericOptimized::do_calibrate_update(
    const int antenna_nr, const Array5D<Matrix2x2<std::complex<float>>>& aterms,
    const Array5D<Matrix2x2<std::complex<float>>>& aterm_derivatives,
    Array4D<double>& hessian, Array3D<double>& gradient,
    Array1D<double>& residual) {
  // Arguments
  auto nr_baselines = m_calibrate_state.nr_baselines;
  auto nr_timesteps = m_calibrate_state.nr_timesteps;
  auto nr_channel_blocks = m_calibrate_state.nr_channel_blocks;
  auto nr_channels_per_block = m_calibrate_state.nr_channels_per_block;
  auto nr_terms = aterm_derivatives.get_c_dim();
  auto subgrid_size = aterms.get_a_dim();
  auto nr_timeslots = aterms.get_d_dim();
  auto nr_stations = aterms.get_c_dim();
  auto nr_polarizations = m_grid->get_z_dim();
  auto grid_size = m_grid->get_y_dim();
  auto image_size = m_cache_state.cell_size * grid_size;
  auto w_step = m_cache_state.w_step;
  auto nr_correlations = 4;

  auto sizeof_aterm_idx =
      auxiliary::sizeof_aterms_indices(nr_baselines, nr_timesteps);
  auto sizeof_visibilities = auxiliary::sizeof_visibilities(
      nr_baselines, nr_timesteps, nr_channels_per_block, nr_correlations);
  auto sizeof_weights = auxiliary::sizeof_weights(
      nr_baselines, nr_timesteps, nr_channels_per_block, nr_correlations);
  auto sizeof_uvw = auxiliary::sizeof_uvw(nr_baselines, nr_timesteps);

  // Performance measurement
  if (antenna_nr == 0) {
    m_report->initialize(nr_channels_per_block, subgrid_size, 0, nr_terms);
  }

  // Start marker
  cu::Marker marker("do_calibrate_update");
  marker.start();

  // Load device
  InstanceCUDA& device = get_device(0);
  const cu::Context& context = device.get_context();

  // Load streams
  cu::Stream& executestream = device.get_execute_stream();
  cu::Stream& htodstream = device.get_htod_stream();
  cu::Stream& dtohstream = device.get_dtoh_stream();

  // Load memory objects
  const unsigned int buffer_nr = 0;

  cu::DeviceMemory& d_wavenumbers = *m_calibrate_state.d_wavenumbers;
  cu::DeviceMemory& d_metadata = *m_calibrate_state.d_metadata[buffer_nr];
  cu::DeviceMemory& d_subgrids = *m_calibrate_state.d_subgrids[buffer_nr];
  cu::DeviceMemory& d_visibilities =
      *m_calibrate_state.d_visibilities[buffer_nr];
  cu::DeviceMemory& d_weights = *m_calibrate_state.d_weights[buffer_nr];
  cu::DeviceMemory& d_uvw = *m_calibrate_state.d_uvw[buffer_nr];
  cu::DeviceMemory& d_sums_x = *m_calibrate_state.d_sums_x;
  cu::DeviceMemory& d_sums_y = *m_calibrate_state.d_sums_y;
  cu::DeviceMemory& d_lmnp = *m_calibrate_state.d_lmnp;
  cu::DeviceMemory& d_aterms_idx =
      *m_calibrate_state.d_aterms_indices[buffer_nr];

  // Allocate additional data structures
  cu::DeviceMemory d_aterms(context, aterms.bytes());
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

  auto uvw_ptr = m_calibrate_state.uvw.data(antenna_nr);
  htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, sizeof_uvw);

  for (unsigned int channel_block = 0; channel_block < nr_channel_blocks;
       channel_block++) {
    auto nr_subgrids =
        m_calibrate_state.plans[antenna_nr][channel_block]->get_nr_subgrids();
    auto sizeof_subgrids =
        auxiliary::sizeof_subgrids(nr_subgrids, subgrid_size, nr_polarizations);
    auto sizeof_metadata = auxiliary::sizeof_metadata(nr_subgrids);

    auto metadata_ptr =
        m_calibrate_state.plans[antenna_nr][channel_block]->get_metadata_ptr();
    auto subgrids_ptr =
        m_calibrate_state.subgrids[antenna_nr][channel_block].data();
    auto aterm_idx_ptr = m_calibrate_state.plans[antenna_nr][channel_block]
                             ->get_aterm_indices_ptr();
    auto visibilities_ptr =
        m_calibrate_state.visibilities.data(antenna_nr, channel_block);
    auto weights_ptr =
        m_calibrate_state.weights.data(antenna_nr, channel_block);

    // Copy input data to device
    htodstream.memcpyHtoDAsync(d_hessian, hessian.data(), hessian.bytes());
    htodstream.memcpyHtoDAsync(d_gradient, gradient.data(), gradient.bytes());
    htodstream.memcpyHtoDAsync(d_residual, &residual, sizeof(double));

    // Copy data to device
    htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, sizeof_metadata);
    htodstream.memcpyHtoDAsync(d_visibilities, visibilities_ptr,
                               sizeof_visibilities);
    htodstream.memcpyHtoDAsync(d_weights, weights_ptr, sizeof_weights);

    htodstream.memcpyHtoDAsync(d_aterms_idx, aterm_idx_ptr, sizeof_aterm_idx);
    htodstream.memcpyHtoDAsync(d_subgrids, subgrids_ptr, sizeof_subgrids);
    htodstream.memcpyHtoDAsync(
        d_wavenumbers, m_calibrate_state.wavenumbers[channel_block].data(),
        m_calibrate_state.wavenumbers[channel_block].bytes());

    // Transpose aterms and aterm derivatives
    const unsigned int nr_aterms = nr_stations * nr_timeslots;
    const unsigned int nr_aterm_derivatives = nr_terms * nr_timeslots;
    Array4D<std::complex<float>> aterms_transposed(nr_aterms, nr_correlations,
                                                   subgrid_size, subgrid_size);
    Array4D<std::complex<float>> aterm_derivatives_transposed(
        nr_aterm_derivatives, nr_correlations, subgrid_size, subgrid_size);
    Array4D<Matrix2x2<std::complex<float>>> aterms_channel_block(
        aterms.data(channel_block), aterms.get_d_dim(), aterms.get_c_dim(),
        aterms.get_b_dim(), aterms.get_a_dim());
    Array4D<Matrix2x2<std::complex<float>>> aterm_derivatives_channel_block(
        aterm_derivatives.data(channel_block), aterm_derivatives.get_d_dim(),
        aterm_derivatives.get_c_dim(), aterm_derivatives.get_b_dim(),
        aterm_derivatives.get_a_dim());
    device.transpose_aterm(nr_polarizations, aterms_channel_block,
                           aterms_transposed);
    device.transpose_aterm(nr_polarizations, aterm_derivatives_channel_block,
                           aterm_derivatives_transposed);
    htodstream.memcpyHtoDAsync(d_aterms, aterms_transposed.data(),
                               aterms_transposed.bytes());
    htodstream.memcpyHtoDAsync(d_aterms_deriv,
                               aterm_derivatives_transposed.data(),
                               aterm_derivatives_transposed.bytes());
    htodstream.record(inputCopied);
    // Run calibration update step
    executestream.waitEvent(inputCopied);
    auto total_nr_timesteps = nr_baselines * nr_timesteps;
    device.launch_calibrate(
        nr_subgrids, grid_size, subgrid_size, image_size, w_step,
        total_nr_timesteps, nr_channels_per_block, nr_stations, nr_terms, d_uvw,
        d_wavenumbers, d_visibilities, d_weights, d_aterms, d_aterms_deriv,
        d_aterms_idx, d_metadata, d_subgrids, d_sums_x, d_sums_y, d_lmnp,
        d_hessian, d_gradient, d_residual);
    executestream.record(executeFinished);

    // Copy output to host
    dtohstream.waitEvent(executeFinished);
    dtohstream.memcpyDtoHAsync(h_hessian.data(), d_hessian, d_hessian.size());
    dtohstream.memcpyDtoHAsync(h_gradient.data(), d_gradient,
                               d_gradient.size());
    dtohstream.memcpyDtoHAsync(h_residual.data(), d_residual,
                               d_residual.size());
    dtohstream.record(outputCopied);

    // Wait for output to finish
    outputCopied.synchronize();
    // Copy output on host
    std::copy_n(static_cast<double*>(h_hessian.data()),
                hessian.size() / nr_channel_blocks,
                hessian.data(channel_block));
    std::copy_n(static_cast<double*>(h_gradient.data()),
                gradient.size() / nr_channel_blocks,
                gradient.data(channel_block));
    std::copy_n(static_cast<double*>(h_residual.data()), 1,
                residual.data(channel_block));
    // Performance reporting
    auto nr_visibilities = nr_timesteps * nr_channels_per_block;
    m_report->update_total(nr_subgrids, nr_timesteps, nr_visibilities);
  }
  // End marker
  marker.end();
}

void GenericOptimized::do_calibrate_finish() {
  // Performance reporting
  auto nr_antennas = m_calibrate_state.plans.size();
  auto nr_channel_blocks = m_calibrate_state.nr_channel_blocks;
  auto total_nr_timesteps = 0;
  auto total_nr_subgrids = 0;
  for (unsigned int antenna_nr = 0; antenna_nr < nr_antennas; antenna_nr++) {
    for (unsigned int channel_block = 0; channel_block < nr_channel_blocks;
         channel_block++) {
      total_nr_timesteps += m_calibrate_state.plans[antenna_nr][channel_block]
                                ->get_nr_timesteps();
      total_nr_subgrids +=
          m_calibrate_state.plans[antenna_nr][channel_block]->get_nr_subgrids();
    }
  }
  m_report->print_total(nr_correlations, total_nr_timesteps, total_nr_subgrids);
  m_report->print_visibilities(auxiliary::name_calibrate);
  m_calibrate_state.d_wavenumbers.reset();
  m_calibrate_state.d_lmnp.reset();
  m_calibrate_state.d_sums_x.reset();
  m_calibrate_state.d_sums_y.reset();
  m_calibrate_state.d_metadata.clear();
  m_calibrate_state.d_subgrids.clear();
  m_calibrate_state.d_visibilities.clear();
  m_calibrate_state.d_weights.clear();
  m_calibrate_state.d_uvw.clear();
  m_calibrate_state.d_aterms_indices.clear();
  get_device(0).free_subgrid_fft();
}

}  // namespace hybrid
}  // namespace proxy
}  // namespace idg
