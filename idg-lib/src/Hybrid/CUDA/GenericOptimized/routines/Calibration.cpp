#include <algorithm>

#include "../GenericOptimized.h"
#include "InstanceCUDA.h"

using namespace idg::proxy::cuda;
using namespace idg::proxy::cpu;
using namespace idg::kernel::cpu;
using namespace idg::kernel::cuda;

namespace idg {
namespace proxy {
namespace hybrid {

void GenericOptimized::do_calibrate_init(
    std::vector<std::vector<std::unique_ptr<Plan>>>&& plans,
    const aocommon::xt::Span<float, 2>& frequencies,
    Tensor<std::complex<float>, 6>&& visibilities, Tensor<float, 6>&& weights,
    Tensor<UVW<float>, 3>&& uvw,
    Tensor<std::pair<unsigned int, unsigned int>, 2>&& baselines,
    const aocommon::xt::Span<float, 2>& taper) {
  std::shared_ptr<InstanceCPU> cpuKernels = cpuProxy->get_kernels();
  cpuKernels->set_report(get_report());

  // Arguments
  const size_t nr_antennas = plans.size();
  const size_t nr_polarizations = get_grid().shape(1);
  const size_t grid_size = get_grid().shape(2);
  assert(get_grid().shape(3) == grid_size);
  const float image_size = m_cache_state.cell_size * grid_size;
  const float w_step = m_cache_state.w_step;
  const size_t subgrid_size = m_cache_state.subgrid_size;
  const size_t nr_channel_blocks = visibilities.Span().shape(1);
  const size_t nr_baselines = visibilities.Span().shape(2);
  const size_t nr_timesteps = visibilities.Span().shape(3);
  const size_t nr_channels_per_block = visibilities.Span().shape(4);
  const size_t nr_correlations = visibilities.Span().shape(5);
  const size_t max_nr_terms = m_calibrate_max_nr_terms;
  const std::array<float, 2>& shift = m_cache_state.shift;

  std::vector<Tensor<float, 1>> wavenumbers;
  for (size_t channel_block = 0; channel_block < nr_channel_blocks;
       channel_block++) {
    auto frequencies_channel_block = aocommon::xt::CreateSpan<float, 1>(
        const_cast<float*>(&frequencies(channel_block, 0)),
        {nr_channels_per_block});
    wavenumbers.push_back(compute_wavenumbers(frequencies_channel_block));
  }

  // Allocate subgrids for all antennas and channel_blocks
  std::vector<std::vector<Tensor<std::complex<float>, 4>>> subgrids(
      nr_antennas);

  // Start performance measurement
  get_report()->initialize();
  pmt::State states[2];
  states[0] = power_meter_->Read();

  // Load device
  InstanceCUDA& device = get_device(0);
  device.set_report(get_report());
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
  m_calibrate_state.d_aterm_indices.clear();

  // Find max number of subgrids
  size_t max_nr_subgrids = 0;
  for (size_t antenna_nr = 0; antenna_nr < nr_antennas; antenna_nr++) {
    for (size_t channel_block = 0; channel_block < nr_channel_blocks;
         channel_block++) {
      const size_t nr_subgrids =
          plans[antenna_nr][channel_block]->get_nr_subgrids();
      if (nr_subgrids > max_nr_subgrids) {
        max_nr_subgrids = nr_subgrids;
      }
    }
  }

  // Allocate device memory
  const size_t sizeof_metadata = auxiliary::sizeof_metadata(max_nr_subgrids);
  const size_t sizeof_subgrids = auxiliary::sizeof_subgrids(
      max_nr_subgrids, subgrid_size, nr_polarizations);
  const size_t sizeof_visibilities = auxiliary::sizeof_visibilities(
      nr_baselines, nr_timesteps, nr_channels_per_block, nr_correlations);
  const size_t sizeof_weights = auxiliary::sizeof_weights(
      nr_baselines, nr_timesteps, nr_channels_per_block, nr_correlations);
  const size_t sizeof_uvw = auxiliary::sizeof_uvw(nr_baselines, nr_timesteps);
  const size_t sizeof_aterm_idx =
      auxiliary::sizeof_aterm_indices(nr_baselines, nr_timesteps);
  const size_t sizeof_wavenumbers =
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
  m_calibrate_state.d_aterm_indices.emplace_back(
      new cu::DeviceMemory(context, sizeof_aterm_idx));

  const size_t buffer_nr = 0;
  cu::DeviceMemory& d_metadata = *m_calibrate_state.d_metadata[buffer_nr];
  cu::DeviceMemory& d_subgrids = *m_calibrate_state.d_subgrids[buffer_nr];

  // Create subgrids for every antenna and channel_block
  for (size_t antenna_nr = 0; antenna_nr < nr_antennas; antenna_nr++) {
    subgrids[antenna_nr].reserve(nr_channel_blocks);
    for (size_t channel_block = 0; channel_block < nr_channel_blocks;
         channel_block++) {
      // Allocate subgrids for current antenna
      const size_t nr_subgrids =
          plans[antenna_nr][channel_block]->get_nr_subgrids();
      Tensor<std::complex<float>, 4> subgrids1 =
          allocate_tensor<std::complex<float>, 4>(
              {nr_subgrids, nr_correlations, subgrid_size, subgrid_size});
      const size_t sizeof_subgrids1 = auxiliary::sizeof_subgrids(
          nr_subgrids, subgrid_size, nr_polarizations);
      const size_t sizeof_metadata1 = auxiliary::sizeof_metadata(nr_subgrids);

      // Get data pointers
      const Metadata* metadata_ptr =
          plans[antenna_nr][channel_block]->get_metadata_ptr();
      std::complex<float>* subgrids_ptr = subgrids1.Span().data();

      // Copy metadata to device
      htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, sizeof_metadata1);

      // Splitter kernel
      if (w_step == 0.0) {
        cpuKernels->run_splitter(nr_subgrids, nr_polarizations, grid_size,
                                 subgrid_size, metadata_ptr, subgrids_ptr,
                                 get_grid().data());
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
              wtile_initialize_set, metadata_ptr, subgrids_ptr,
              get_grid().data());
        }
      } else {
        cpuKernels->run_splitter_wstack(nr_subgrids, nr_polarizations,
                                        grid_size, subgrid_size, metadata_ptr,
                                        subgrids_ptr, get_grid().data());
      }

      // FFT kernel
      cpuKernels->run_subgrid_fft(grid_size, subgrid_size,
                                  nr_subgrids * nr_polarizations, subgrids_ptr,
                                  CUFFT_FORWARD);

      // Apply taper
      for (size_t i = 0; i < nr_subgrids; i++) {
        for (size_t pol = 0; pol < nr_correlations; pol++) {
          for (size_t j = 0; j < subgrid_size; j++) {
            for (size_t k = 0; k < subgrid_size; k++) {
              // Apply FFT shift (swapping corner and centre) to subgrid indices
              const size_t y = (j + (subgrid_size / 2)) % subgrid_size;
              const size_t x = (k + (subgrid_size / 2)) % subgrid_size;
              subgrids1.Span()(i, pol, y, x) *= taper(j, k);
            }
          }
        }
      }

      subgrids[antenna_nr].push_back(std::move(subgrids1));

    }  // end for channel_blocks
  }    // end for antennas

  // End performance measurement
  states[1] = power_meter_->Read();
  get_report()->update(Report::host, states[0], states[1]);
  get_report()->print_total(0, 0, 0);

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
  const size_t sizeof_lmnp =
      max_nr_subgrids * subgrid_size * subgrid_size * 4 * sizeof(float);
  m_calibrate_state.d_lmnp.reset(new cu::DeviceMemory(context, sizeof_lmnp));

  // Allocate memory for sums (horizontal and vertical)
  const size_t total_nr_timesteps = nr_baselines * nr_timesteps;
  const size_t sizeof_sums = max_nr_terms * nr_correlations *
                             total_nr_timesteps * nr_channels_per_block *
                             sizeof(std::complex<float>);
  m_calibrate_state.d_sums_x.reset(new cu::DeviceMemory(context, sizeof_sums));
  m_calibrate_state.d_sums_y.reset(new cu::DeviceMemory(context, sizeof_sums));
}

void GenericOptimized::do_calibrate_update(
    const int antenna_nr,
    const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 5>& aterms,
    const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 5>&
        aterm_derivatives,
    aocommon::xt::Span<double, 4>& hessian,
    aocommon::xt::Span<double, 3>& gradient,
    aocommon::xt::Span<double, 1>& residual) {
  // Arguments
  const size_t nr_baselines = m_calibrate_state.nr_baselines;
  const size_t nr_timesteps = m_calibrate_state.nr_timesteps;
  const size_t nr_channel_blocks = m_calibrate_state.nr_channel_blocks;
  const size_t nr_channels_per_block = m_calibrate_state.nr_channels_per_block;
  const size_t nr_terms = aterm_derivatives.shape(2);
  const size_t subgrid_size = aterms.shape(4);
  assert(subgrid_size == aterms.shape(3));
  const size_t nr_timeslots = aterms.shape(1);
  const size_t nr_stations = aterms.shape(2);
  const size_t nr_polarizations = get_grid().shape(1);
  const size_t grid_size = get_grid().shape(2);
  assert(get_grid().shape(3) == grid_size);
  const float image_size = m_cache_state.cell_size * grid_size;
  const float w_step = m_cache_state.w_step;
  const size_t nr_correlations = 4;

  const size_t sizeof_aterm_idx =
      auxiliary::sizeof_aterm_indices(nr_baselines, nr_timesteps);
  const size_t sizeof_visibilities = auxiliary::sizeof_visibilities(
      nr_baselines, nr_timesteps, nr_channels_per_block, nr_correlations);
  const size_t sizeof_weights = auxiliary::sizeof_weights(
      nr_baselines, nr_timesteps, nr_channels_per_block, nr_correlations);
  const size_t sizeof_uvw = auxiliary::sizeof_uvw(nr_baselines, nr_timesteps);

  // Performance measurement
  if (antenna_nr == 0) {
    get_report()->initialize(nr_channels_per_block, subgrid_size, 0, nr_terms);
  }

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
  const size_t buffer_nr = 0;
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
  cu::DeviceMemory& d_aterm_indices =
      *m_calibrate_state.d_aterm_indices[buffer_nr];

  // Allocate additional data structures
  const size_t sizeof_aterms = aterms.size() * sizeof(*aterms.data());
  const size_t sizeof_aterm_derivatives =
      aterm_derivatives.size() * sizeof(*aterm_derivatives.data());
  const size_t sizeof_hessian = hessian.size() * sizeof(*hessian.data());
  const size_t sizeof_gradient = gradient.size() * sizeof(*gradient.data());
  cu::DeviceMemory d_aterms(context, sizeof_aterms);
  cu::DeviceMemory d_aterm_derivatives(context, sizeof_aterm_derivatives);
  cu::DeviceMemory d_hessian(context, sizeof_hessian);
  cu::DeviceMemory d_gradient(context, sizeof_gradient);
  cu::DeviceMemory d_residual(context, sizeof(double));
  cu::HostMemory h_hessian(context, sizeof_hessian);
  cu::HostMemory h_gradient(context, sizeof_gradient);
  cu::HostMemory h_residual(context, sizeof(double));

  // Events
  cu::Event inputCopied(context), executeFinished(context),
      outputCopied(context);

  UVW<float>* uvw_ptr = &m_calibrate_state.uvw.Span()(antenna_nr, 0, 0);
  htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, sizeof_uvw);

  for (size_t channel_block = 0; channel_block < nr_channel_blocks;
       channel_block++) {
    const size_t nr_subgrids =
        m_calibrate_state.plans[antenna_nr][channel_block]->get_nr_subgrids();
    const size_t sizeof_subgrids =
        auxiliary::sizeof_subgrids(nr_subgrids, subgrid_size, nr_polarizations);
    const size_t sizeof_metadata = auxiliary::sizeof_metadata(nr_subgrids);
    const size_t sizeof_wavenumbers =
        m_calibrate_state.wavenumbers[channel_block].Span().size() *
        sizeof(*m_calibrate_state.wavenumbers[channel_block].Span().data());

    const Metadata* metadata_ptr =
        m_calibrate_state.plans[antenna_nr][channel_block]->get_metadata_ptr();
    std::complex<float>* subgrids_ptr =
        m_calibrate_state.subgrids[antenna_nr][channel_block].Span().data();
    const unsigned int* aterm_idx_ptr =
        m_calibrate_state.plans[antenna_nr][channel_block]
            ->get_aterm_indices_ptr();
    const std::complex<float>* visibilities_ptr =
        &m_calibrate_state.visibilities.Span()(antenna_nr, channel_block, 0, 0,
                                               0, 0);
    const float* weights_ptr = &m_calibrate_state.weights.Span()(
        antenna_nr, channel_block, 0, 0, 0, 0);

    // Copy input data to device
    htodstream.memcpyHtoDAsync(d_hessian, hessian.data(), sizeof_hessian);
    htodstream.memcpyHtoDAsync(d_gradient, gradient.data(), sizeof_gradient);
    htodstream.memcpyHtoDAsync(d_residual, &residual, sizeof(double));

    // Copy data to device
    htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, sizeof_metadata);
    htodstream.memcpyHtoDAsync(d_visibilities, visibilities_ptr,
                               sizeof_visibilities);
    htodstream.memcpyHtoDAsync(d_weights, weights_ptr, sizeof_weights);

    htodstream.memcpyHtoDAsync(d_aterm_indices, aterm_idx_ptr,
                               sizeof_aterm_idx);
    htodstream.memcpyHtoDAsync(d_subgrids, subgrids_ptr, sizeof_subgrids);
    htodstream.memcpyHtoDAsync(
        d_wavenumbers,
        m_calibrate_state.wavenumbers[channel_block].Span().data(),
        sizeof_wavenumbers);

    // Transpose aterms and aterm derivatives
    const size_t nr_aterms = nr_stations * nr_timeslots;
    const size_t nr_aterm_derivatives = nr_terms * nr_timeslots;
    Tensor<std::complex<float>, 4> aterms_transposed =
        allocate_tensor<std::complex<float>, 4>(
            {nr_aterms, nr_correlations, subgrid_size, subgrid_size});
    Tensor<std::complex<float>, 4> aterm_derivatives_transposed =
        allocate_tensor<std::complex<float>, 4>({nr_aterm_derivatives,
                                                 nr_correlations, subgrid_size,
                                                 subgrid_size});
    using Aterm = Matrix2x2<std::complex<float>>;
    auto aterms_channel_block = aocommon::xt::CreateSpan<Aterm, 4>(
        const_cast<Matrix2x2<std::complex<float>>*>(
            &aterms(channel_block, 0, 0, 0, 0)),
        {aterms.shape(1), aterms.shape(2), aterms.shape(3), aterms.shape(4)});
    auto aterm_derivatives_channel_block = aocommon::xt::CreateSpan<Aterm, 4>(
        const_cast<Matrix2x2<std::complex<float>>*>(
            &aterm_derivatives(channel_block, 0, 0, 0, 0)),
        {aterm_derivatives.shape(1), aterm_derivatives.shape(2),
         aterm_derivatives.shape(3), aterm_derivatives.shape(4)});
    const size_t sizeof_aterms_channel_block =
        aterms_channel_block.size() * sizeof(*aterms_channel_block.data());
    const size_t sizeof_aterm_derivatives_channel_block =
        aterm_derivatives_channel_block.size() *
        sizeof(*aterm_derivatives_channel_block.data());
    device.transpose_aterm(nr_polarizations, aterms_channel_block,
                           aterms_transposed.Span());
    device.transpose_aterm(nr_polarizations, aterm_derivatives_channel_block,
                           aterm_derivatives_transposed.Span());
    htodstream.memcpyHtoDAsync(d_aterms, aterms_transposed.Span().data(),
                               sizeof_aterms_channel_block);
    htodstream.memcpyHtoDAsync(d_aterm_derivatives,
                               aterm_derivatives_transposed.Span().data(),
                               sizeof_aterm_derivatives_channel_block);
    htodstream.record(inputCopied);

    // Run calibration update step
    executestream.waitEvent(inputCopied);
    const size_t total_nr_timesteps = nr_baselines * nr_timesteps;
    device.launch_calibrate(
        nr_subgrids, grid_size, subgrid_size, image_size, w_step,
        total_nr_timesteps, nr_channels_per_block, nr_stations, nr_terms, d_uvw,
        d_wavenumbers, d_visibilities, d_weights, d_aterms, d_aterm_derivatives,
        d_aterm_indices, d_metadata, d_subgrids, d_sums_x, d_sums_y, d_lmnp,
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
                &hessian(channel_block, 0, 0, 0));
    std::copy_n(static_cast<double*>(h_gradient.data()),
                gradient.size() / nr_channel_blocks,
                &gradient(channel_block, 0, 0));
    std::copy_n(static_cast<double*>(h_residual.data()), 1,
                &residual(channel_block));

    // Performance reporting
    auto nr_visibilities = nr_timesteps * nr_channels_per_block;
    get_report()->update_total(nr_subgrids, nr_timesteps, nr_visibilities);
  }

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
  get_report()->print_total(nr_correlations, total_nr_timesteps,
                            total_nr_subgrids);
  get_report()->print_visibilities(auxiliary::name_calibrate);
  m_calibrate_state.d_wavenumbers.reset();
  m_calibrate_state.d_lmnp.reset();
  m_calibrate_state.d_sums_x.reset();
  m_calibrate_state.d_sums_y.reset();
  m_calibrate_state.d_metadata.clear();
  m_calibrate_state.d_subgrids.clear();
  m_calibrate_state.d_visibilities.clear();
  m_calibrate_state.d_weights.clear();
  m_calibrate_state.d_uvw.clear();
  m_calibrate_state.d_aterm_indices.clear();
  get_device(0).free_subgrid_fft();
}

}  // namespace hybrid
}  // namespace proxy
}  // namespace idg
