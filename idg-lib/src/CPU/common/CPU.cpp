// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <vector>
#include <memory>
#include <climits>

#include "fftw3.h"

#include "CPU.h"

//#define DEBUG_COMPUTE_JOBSIZE

using namespace idg::kernel;

namespace idg {
namespace proxy {
namespace cpu {

// Constructor
CPU::CPU() : power_meter_(pmt::get_power_meter(pmt::sensor_host)) {
#if defined(DEBUG)
  std::cout << "CPU::" << __func__ << std::endl;
#endif
}

// Destructor
CPU::~CPU() {
#if defined(DEBUG)
  std::cout << "CPU::" << __func__ << std::endl;
#endif

  // Deallocate FFTWs internally allocated memory
  fftwf_cleanup();
}

std::unique_ptr<auxiliary::Memory> CPU::allocate_memory(size_t bytes) {
  return std::unique_ptr<auxiliary::Memory>(
      new auxiliary::AlignedMemory(bytes));
}

std::unique_ptr<Plan> CPU::make_plan(
    const int kernel_size, const aocommon::xt::Span<float, 1>& frequencies,
    const aocommon::xt::Span<UVW<float>, 2>& uvw,
    const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
        baselines,
    const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
    Plan::Options options) {
  if (supports_wtiling() && m_cache_state.w_step != 0.0 &&
      m_wtiles.get_wtile_buffer_size()) {
    options.w_step = m_cache_state.w_step;
    options.nr_w_layers = INT_MAX;
    const size_t grid_size = get_grid().shape(2);
    assert(get_grid().shape(3) == grid_size);
    return std::unique_ptr<Plan>(
        new Plan(kernel_size, m_cache_state.subgrid_size, grid_size,
                 m_cache_state.cell_size, m_cache_state.shift, frequencies, uvw,
                 baselines, aterm_offsets, m_wtiles, options));
  } else {
    return Proxy::make_plan(kernel_size, frequencies, uvw, baselines,
                            aterm_offsets, options);
  }
}

void CPU::init_cache(int subgrid_size, float cell_size, float w_step,
                     const std::array<float, 2>& shift) {
  Proxy::init_cache(subgrid_size, cell_size, w_step, shift);
  const int nr_polarizations = get_grid().shape(1);
  const size_t grid_size = get_grid().shape(2);
  assert(get_grid().shape(3) == grid_size);
  const int nr_wtiles =
      m_kernels->init_wtiles(nr_polarizations, grid_size, subgrid_size);
  m_wtiles = WTiles(nr_wtiles, kernel::cpu::InstanceCPU::kWTileSize);
}

aocommon::xt::Span<std::complex<float>, 4>& CPU::get_final_grid() {
  // flush all pending Wtiles
  WTileUpdateInfo wtile_flush_info = m_wtiles.clear();
  if (wtile_flush_info.wtile_ids.size()) {
    const size_t nr_polarizations = get_grid().shape(1);
    const size_t grid_size = get_grid().shape(2);
    assert(get_grid().shape(3) == grid_size);
    const float image_size = grid_size * m_cache_state.cell_size;
    const size_t subgrid_size = m_cache_state.subgrid_size;
    const float w_step = m_cache_state.w_step;
    auto& shift = m_cache_state.shift;
    pmt::State states[2];
    get_report()->initialize(0, subgrid_size, grid_size);
    states[0] = power_meter_->Read();
    m_kernels->run_adder_tiles_to_grid(
        nr_polarizations, grid_size, subgrid_size, image_size, w_step,
        shift.data(), wtile_flush_info.wtile_ids.size(),
        wtile_flush_info.wtile_ids.data(),
        wtile_flush_info.wtile_coordinates.data(), get_grid().data());
    states[1] = power_meter_->Read();
    get_report()->update(Report::wtiling_forward, states[0], states[1]);
    get_report()->print_total(nr_correlations);
  }
  return get_grid();
}

unsigned int CPU::compute_jobsize(const Plan& plan,
                                  const unsigned int nr_timesteps,
                                  const unsigned int nr_channels,
                                  const unsigned int nr_correlations,
                                  const unsigned int nr_polarizations,
                                  const unsigned int subgrid_size) {
  auto nr_baselines = plan.get_nr_baselines();
  auto jobsize = nr_baselines;
  auto sizeof_visibilities = auxiliary::sizeof_visibilities(
      nr_baselines, nr_timesteps, nr_channels, nr_correlations);

  // Make sure that every job will fit in memory
  do {
    // Determine the maximum number of subgrids for this jobsize
    auto max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);

    // Determine the size of the subgrids for this jobsize
    auto sizeof_subgrids = auxiliary::sizeof_subgrids(
        max_nr_subgrids, subgrid_size, nr_polarizations);

#if defined(DEBUG_COMPUTE_JOBSIZE)
    std::clog << "size of subgrids: " << sizeof_subgrids << std::endl;
#endif

    // Determine the amount of free memory
    auto free_memory = auxiliary::get_free_memory();  // Mb
    free_memory *= 1024 * 1024;                       // Byte

    // Limit the amount of memory used for subgrids
    free_memory *= m_fraction_memory_subgrids;

    // Determine whether to proceed with the current jobsize
    if (sizeof_subgrids < sizeof_visibilities &&
        sizeof_subgrids < free_memory &&
        sizeof_subgrids < m_max_bytes_subgrids) {
      break;
    }

    // Reduce jobsize
    jobsize *= 0.8;
  } while (jobsize > 1);

#if defined(DEBUG_COMPUTE_JOBSIZE)
  std::clog << "jobsize: " << jobsize << std::endl;
#endif

  return std::max(1, jobsize);
}

/*
    High level routines
*/
void CPU::do_gridding(
    const Plan& plan, const aocommon::xt::Span<float, 1>& frequencies,
    const aocommon::xt::Span<std::complex<float>, 4>& visibilities,
    const aocommon::xt::Span<UVW<float>, 2>& uvw,
    const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
        baselines,
    const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
    const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
    const aocommon::xt::Span<float, 2>& taper) {
#if defined(DEBUG)
  std::cout << __func__ << std::endl;
#endif

  m_kernels->set_report(get_report());

  Tensor<float, 1> wavenumbers = compute_wavenumbers(frequencies);

  // Arguments
  const size_t nr_baselines = visibilities.shape(0);
  const size_t nr_timesteps = visibilities.shape(1);
  const size_t nr_channels = visibilities.shape(2);
  const size_t nr_correlations = visibilities.shape(3);
  const size_t nr_polarizations = get_grid().shape(1);
  const size_t grid_size = get_grid().shape(2);
  assert(get_grid().shape(3) == grid_size);
  const size_t subgrid_size = plan.get_subgrid_size();
  const std::array<float, 2>& shift = plan.get_shift();
  const float w_step = plan.get_w_step();
  const float image_size = plan.get_cell_size() * grid_size;
  const size_t nr_stations = aterms.shape(1);

  WTileUpdateSet wtile_flush_set = plan.get_wtile_flush_set();

  try {
    auto jobsize =
        compute_jobsize(plan, nr_timesteps, nr_channels, nr_correlations,
                        nr_polarizations, subgrid_size);

    // Allocate memory for subgrids
    const size_t max_nr_subgrids =
        plan.get_max_nr_subgrids(0, nr_baselines, jobsize);
    Tensor<std::complex<float>, 4> subgrids =
        allocate_tensor<std::complex<float>, 4>(
            {max_nr_subgrids, nr_correlations, subgrid_size, subgrid_size});

    // Performance measurement
    get_report()->initialize(nr_channels, subgrid_size, grid_size);
    pmt::State states[2];
    states[0] = power_meter_->Read();

    // Start gridder
    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
      unsigned int first_bl, last_bl, current_nr_baselines;
      plan.initialize_job(nr_baselines, jobsize, bl, &first_bl, &last_bl,
                          &current_nr_baselines);
      if (current_nr_baselines == 0) continue;

      // Initialize iteration
      auto current_nr_subgrids =
          plan.get_nr_subgrids(first_bl, current_nr_baselines);
      const float* shift_ptr = shift.data();
      const float* wavenumbers_ptr = wavenumbers.Span().data();
      auto* taper_ptr = taper.data();
      auto* aterm_ptr =
          reinterpret_cast<const std::complex<float>*>(aterms.data());
      const unsigned int* aterm_idx_ptr = plan.get_aterm_indices_ptr();
      auto* avg_aterm_ptr = m_avg_aterm_correction.size()
                                ? m_avg_aterm_correction.data()
                                : nullptr;
      auto* metadata_ptr = plan.get_metadata_ptr(first_bl);
      const UVW<float>* uvw_ptr = uvw.data();
      const std::complex<float>* visibilities_ptr = visibilities.data();
      std::complex<float>* subgrids_ptr = subgrids.Span().data();
      std::complex<float>* grid_ptr = get_grid().data();

      // Gridder kernel
      m_kernels->run_gridder(
          current_nr_subgrids, nr_polarizations, grid_size, subgrid_size,
          image_size, w_step, shift_ptr, nr_channels, nr_correlations,
          nr_stations, uvw_ptr, wavenumbers_ptr, visibilities_ptr, taper_ptr,
          aterm_ptr, aterm_idx_ptr, avg_aterm_ptr, metadata_ptr, subgrids_ptr);

      // FFT kernel
      m_kernels->run_subgrid_fft(grid_size, subgrid_size,
                                 current_nr_subgrids * nr_correlations,
                                 subgrids_ptr, FFTW_BACKWARD);

      // Adder kernel
      if (plan.get_use_wtiles()) {
        auto subgrid_offset = plan.get_subgrid_offset(bl);
        m_kernels->run_adder_wtiles(current_nr_subgrids, nr_polarizations,
                                    grid_size, subgrid_size, image_size, w_step,
                                    shift_ptr, subgrid_offset, wtile_flush_set,
                                    metadata_ptr, subgrids_ptr, grid_ptr);
      } else if (w_step != 0.0) {
        m_kernels->run_adder_wstack(current_nr_subgrids, nr_polarizations,
                                    grid_size, subgrid_size, metadata_ptr,
                                    subgrids_ptr, grid_ptr);
      } else {
        m_kernels->run_adder(current_nr_subgrids, nr_polarizations, grid_size,
                             subgrid_size, metadata_ptr, subgrids_ptr,
                             grid_ptr);
      }

      // Performance reporting
      auto current_nr_timesteps =
          plan.get_nr_timesteps(first_bl, current_nr_baselines);
      get_report()->print(nr_correlations, current_nr_timesteps,
                          current_nr_subgrids);
    }  // end for bl

    states[1] = power_meter_->Read();
    get_report()->update(Report::host, states[0], states[1]);

    // Performance report
    auto total_nr_subgrids = plan.get_nr_subgrids();
    auto total_nr_timesteps = plan.get_nr_timesteps();
    get_report()->print_total(nr_correlations, total_nr_timesteps,
                              total_nr_subgrids);
    auto total_nr_visibilities = plan.get_nr_visibilities();
    get_report()->print_visibilities(auxiliary::name_gridding,
                                     total_nr_visibilities);

  } catch (const std::invalid_argument& e) {
    std::cerr << __func__ << ": invalid argument: " << e.what() << std::endl;
    exit(1);
  } catch (const std::exception& e) {
    std::cerr << __func__ << ": caught exception: " << e.what() << std::endl;
    exit(2);
  } catch (...) {
    std::cerr << __func__ << ": caught unknown exception" << std::endl;
    exit(3);
  }
}  // end gridding

void CPU::do_degridding(
    const Plan& plan, const aocommon::xt::Span<float, 1>& frequencies,
    aocommon::xt::Span<std::complex<float>, 4>& visibilities,
    const aocommon::xt::Span<UVW<float>, 2>& uvw,
    const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
        baselines,
    const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
    const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
    const aocommon::xt::Span<float, 2>& taper) {
#if defined(DEBUG)
  std::cout << __func__ << std::endl;
#endif

  m_kernels->set_report(get_report());

  Tensor<float, 1> wavenumbers = compute_wavenumbers(frequencies);

  // Arguments
  const size_t nr_baselines = visibilities.shape(0);
  const size_t nr_timesteps = visibilities.shape(1);
  const size_t nr_channels = visibilities.shape(2);
  const size_t nr_correlations = visibilities.shape(3);
  const size_t nr_polarizations = get_grid().shape(1);
  const size_t grid_size = get_grid().shape(2);
  assert(get_grid().shape(3) == grid_size);
  const float image_size = plan.get_cell_size() * grid_size;
  const size_t subgrid_size = plan.get_subgrid_size();
  const float w_step = plan.get_w_step();
  const size_t nr_stations = aterms.shape(1);
  const std::array<float, 2>& shift = plan.get_shift();

  WTileUpdateSet wtile_initialize_set = plan.get_wtile_initialize_set();

  try {
    auto jobsize =
        compute_jobsize(plan, nr_timesteps, nr_channels, nr_correlations,
                        nr_polarizations, subgrid_size);

    // Allocate memory for subgrids
    const size_t max_nr_subgrids =
        plan.get_max_nr_subgrids(0, nr_baselines, jobsize);
    Tensor<std::complex<float>, 4> subgrids =
        allocate_tensor<std::complex<float>, 4>(
            {max_nr_subgrids, nr_correlations, subgrid_size, subgrid_size});

    // Performance measurement
    get_report()->initialize(nr_channels, subgrid_size, grid_size);
    pmt::State states[2];
    states[0] = power_meter_->Read();

    // Run subroutines
    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
      unsigned int first_bl, last_bl, current_nr_baselines;
      plan.initialize_job(nr_baselines, jobsize, bl, &first_bl, &last_bl,
                          &current_nr_baselines);
      if (current_nr_baselines == 0) continue;

      // Initialize iteration
      const size_t current_nr_subgrids =
          plan.get_nr_subgrids(first_bl, current_nr_baselines);
      const float* shift_ptr = shift.data();
      const float* wavenumbers_ptr = wavenumbers.Span().data();
      auto* taper_ptr = taper.data();
      auto* aterm_ptr =
          reinterpret_cast<const std::complex<float>*>(aterms.data());
      const unsigned int* aterm_idx_ptr = plan.get_aterm_indices_ptr();
      auto* metadata_ptr = plan.get_metadata_ptr(first_bl);
      const UVW<float>* uvw_ptr = uvw.data();
      std::complex<float>* visibilities_ptr = visibilities.data();
      std::complex<float>* subgrids_ptr = subgrids.Span().data();
      const std::complex<float>* grid_ptr = get_grid().data();

      // Splitter kernel
      if (plan.get_use_wtiles()) {
        auto subgrid_offset = plan.get_subgrid_offset(bl);
        m_kernels->run_splitter_wtiles(
            current_nr_subgrids, nr_polarizations, grid_size, subgrid_size,
            image_size, w_step, shift_ptr, subgrid_offset, wtile_initialize_set,
            metadata_ptr, subgrids_ptr, grid_ptr);
      } else if (w_step != 0.0) {
        m_kernels->run_splitter_wstack(current_nr_subgrids, nr_polarizations,
                                       grid_size, subgrid_size, metadata_ptr,
                                       subgrids_ptr, grid_ptr);
      } else {
        m_kernels->run_splitter(current_nr_subgrids, nr_polarizations,
                                grid_size, subgrid_size, metadata_ptr,
                                subgrids_ptr, grid_ptr);
      }

      // FFT kernel
      m_kernels->run_subgrid_fft(grid_size, subgrid_size,
                                 current_nr_subgrids * nr_correlations,
                                 subgrids_ptr, FFTW_FORWARD);

      // Degridder kernel
      m_kernels->run_degridder(
          current_nr_subgrids, nr_polarizations, grid_size, subgrid_size,
          image_size, w_step, shift_ptr, nr_channels, nr_correlations,
          nr_stations, uvw_ptr, wavenumbers_ptr, visibilities_ptr, taper_ptr,
          aterm_ptr, aterm_idx_ptr, metadata_ptr, subgrids_ptr);

      // Performance reporting
      auto current_nr_timesteps =
          plan.get_nr_timesteps(first_bl, current_nr_baselines);
      get_report()->print(nr_correlations, current_nr_timesteps,
                          current_nr_subgrids);
    }  // end for bl

    states[1] = power_meter_->Read();
    get_report()->update(Report::host, states[0], states[1]);

    // Report performance
    auto total_nr_subgrids = plan.get_nr_subgrids();
    auto total_nr_timesteps = plan.get_nr_timesteps();
    get_report()->print_total(nr_correlations, total_nr_timesteps,
                              total_nr_subgrids);
    auto total_nr_visibilities = plan.get_nr_visibilities();
    get_report()->print_visibilities(auxiliary::name_degridding,
                                     total_nr_visibilities);

  } catch (const std::invalid_argument& e) {
    std::cerr << __func__ << ": invalid argument: " << e.what() << std::endl;
    exit(1);
  } catch (const std::exception& e) {
    std::cerr << __func__ << ": caught exception: " << e.what() << std::endl;
    exit(2);
  } catch (...) {
    std::cerr << __func__ << ": caught unknown exception" << std::endl;
    exit(3);
  }
}  // end degridding

void CPU::do_calibrate_init(
    std::vector<std::vector<std::unique_ptr<Plan>>>&& plans,
    const aocommon::xt::Span<float, 2>& frequencies,
    Tensor<std::complex<float>, 6>&& visibilities, Tensor<float, 6>&& weights,
    Tensor<UVW<float>, 3>&& uvw,
    Tensor<std::pair<unsigned int, unsigned int>, 2>&& baselines,
    const aocommon::xt::Span<float, 2>& taper) {
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
  const size_t nr_channels = visibilities.Span().shape(4);
  const size_t nr_correlations = visibilities.Span().shape(5);
  assert(nr_correlations == 4);

  if (nr_channel_blocks > 1) {
    throw std::runtime_error(
        "nr_channel_blocks>1 in calibration is not supported by CPU Proxy.");
  }

  auto frequencies_span = aocommon::xt::CreateSpan<float, 1>(
      const_cast<float*>(frequencies.data()), {nr_channels});
  Tensor<float, 1> wavenumbers = compute_wavenumbers(frequencies_span);

  // Allocate subgrids for all antennas
  std::vector<Tensor<std::complex<float>, 4>> subgrids;
  subgrids.reserve(nr_antennas);

  // Allocate phasors for all antennas
  std::vector<Tensor<std::complex<float>, 4>> phasors;
  phasors.reserve(nr_antennas);

  std::vector<int> max_nr_timesteps;
  max_nr_timesteps.reserve(nr_antennas);

  // Start performance measurement
  get_report()->initialize();
  pmt::State states[2];
  states[0] = power_meter_->Read();

  // Create subgrids for every antenna
  for (size_t antenna_nr = 0; antenna_nr < nr_antennas; antenna_nr++) {
    // Allocate subgrids for current antenna
    const size_t nr_subgrids = plans[antenna_nr][0]->get_nr_subgrids();
    Tensor<std::complex<float>, 4> subgrids_tensor =
        allocate_tensor<std::complex<float>, 4>(
            {nr_subgrids, nr_correlations,
             static_cast<size_t>(m_cache_state.subgrid_size),
             static_cast<size_t>(m_cache_state.subgrid_size)});
    aocommon::xt::Span<std::complex<float>, 4> subgrids_ =
        subgrids_tensor.Span();

    WTileUpdateSet wtile_initialize_set =
        plans[antenna_nr][0]->get_wtile_initialize_set();

    // Get data pointers
    float* shift_ptr = m_cache_state.shift.data();
    const Metadata* metadata_ptr = plans[antenna_nr][0]->get_metadata_ptr();
    std::complex<float>* subgrids_ptr = subgrids_.data();
    const std::complex<float>* grid_ptr = get_grid().data();

    // Splitter kernel
    if (w_step == 0.0) {
      m_kernels->run_splitter(nr_subgrids, nr_polarizations, grid_size,
                              subgrid_size, metadata_ptr, subgrids_ptr,
                              grid_ptr);
    } else if (plans[antenna_nr][0]->get_use_wtiles()) {
      m_kernels->run_splitter_wtiles(
          nr_subgrids, nr_polarizations, grid_size, subgrid_size, image_size,
          w_step, shift_ptr, 0 /* subgrid_offset */, wtile_initialize_set,
          metadata_ptr, subgrids_ptr, grid_ptr);
    } else {
      m_kernels->run_splitter_wstack(nr_subgrids, nr_polarizations, grid_size,
                                     subgrid_size, metadata_ptr, subgrids_ptr,
                                     grid_ptr);
    }

    // FFT kernel
    m_kernels->run_subgrid_fft(grid_size, subgrid_size,
                               nr_subgrids * nr_correlations, subgrids_ptr,
                               FFTW_FORWARD);

    // Apply taper
    for (size_t i = 0; i < nr_subgrids; i++) {
      for (size_t pol = 0; pol < nr_polarizations; pol++) {
        for (size_t j = 0; j < subgrid_size; j++) {
          for (size_t k = 0; k < subgrid_size; k++) {
            const size_t y = (j + (subgrid_size / 2)) % subgrid_size;
            const size_t x = (k + (subgrid_size / 2)) % subgrid_size;
            subgrids_(i, pol, y, x) *= taper(j, k);
          }
        }
      }
    }

    // Store subgrids for current antenna
    subgrids.push_back(std::move(subgrids_tensor));

    // Get max number of timesteps for any subgrid
    const size_t max_nr_timesteps_ =
        plans[antenna_nr][0]->get_max_nr_timesteps_subgrid();
    max_nr_timesteps.push_back(max_nr_timesteps_);

    // Allocate phasors for current antenna
    Tensor<std::complex<float>, 4> phasors_tensor =
        allocate_tensor<std::complex<float>, 4>(
            {nr_subgrids * max_nr_timesteps_, nr_channels, subgrid_size,
             subgrid_size});
    aocommon::xt::Span<std::complex<float>, 4> phasors_ = phasors_tensor.Span();
    phasors_.fill(std::complex<float>(0, 0));

    // Compute phasors
    const idg::UVW<float>* uvw_ptr = &uvw.Span()(antenna_nr, 0, 0);
    m_kernels->run_calibrate_phasor(
        nr_subgrids, grid_size, subgrid_size, image_size, w_step, shift_ptr,
        max_nr_timesteps_, nr_channels, uvw_ptr, wavenumbers.Span().data(),
        metadata_ptr, phasors_.data());

    // Store phasors for current antenna
    phasors.push_back(std::move(phasors_tensor));
  }  // end for antennas

  // End performance measurement
  states[1] = power_meter_->Read();
  get_report()->update<Report::ID::host>(states[0], states[1]);
  get_report()->print_total(nr_correlations);

  // Set calibration state member variables
  m_calibrate_state = {.plans = std::move(plans),
                       .nr_baselines = nr_baselines,
                       .nr_timesteps = nr_timesteps,
                       .nr_channels = nr_channels,
                       .wavenumbers = std::move(wavenumbers),
                       .visibilities = std::move(visibilities),
                       .weights = std::move(weights),
                       .uvw = std::move(uvw),
                       .baselines = std::move(baselines),
                       .subgrids = std::move(subgrids),
                       .phasors = std::move(phasors),
                       .max_nr_timesteps = std::move(max_nr_timesteps)};
}

void CPU::do_calibrate_update(
    const int antenna_nr,
    const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 5>& aterms,
    const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 5>&
        aterm_derivatives,
    aocommon::xt::Span<double, 4>& hessian,
    aocommon::xt::Span<double, 3>& gradient,
    aocommon::xt::Span<double, 1>& residual) {
  if (m_calibrate_state.plans.empty()) {
    throw std::runtime_error("Calibration was not initialized. Can not update");
  }

  // Arguments
  const size_t nr_subgrids =
      m_calibrate_state.plans[antenna_nr][0]->get_nr_subgrids();
  const size_t nr_channels = m_calibrate_state.wavenumbers.Span().size();
  const size_t nr_terms = aterm_derivatives.shape(2);
  const size_t subgrid_size = aterms.shape(4);
  assert(subgrid_size == aterms.shape(3));
  const size_t nr_stations = aterms.shape(2);
  const size_t nr_timeslots = aterms.shape(1);
  const size_t nr_polarizations = get_grid().shape(1);
  const size_t grid_size = get_grid().shape(2);
  assert(get_grid().shape(3) == grid_size);
  const float image_size = grid_size * m_cache_state.cell_size;
  const float w_step = m_cache_state.w_step;

  // Performance measurement
  if (antenna_nr == 0) {
    get_report()->initialize(nr_channels, subgrid_size, 0, nr_terms);
  }

  // Data pointers
  float* shift_ptr = m_cache_state.shift.data();
  float* wavenumbers_ptr = m_calibrate_state.wavenumbers.Span().data();
  const std::complex<float>* aterm_ptr =
      reinterpret_cast<const std::complex<float>*>(aterms.data());
  const std::complex<float>* aterm_derivative_ptr =
      reinterpret_cast<const std::complex<float>*>(aterm_derivatives.data());
  const unsigned int* aterm_idx_ptr =
      m_calibrate_state.plans[antenna_nr][0]->get_aterm_indices_ptr();
  const Metadata* metadata_ptr =
      m_calibrate_state.plans[antenna_nr][0]->get_metadata_ptr();
  UVW<float>* uvw_ptr = &m_calibrate_state.uvw.Span()(antenna_nr, 0, 0);
  std::complex<float>* visibilities_ptr =
      reinterpret_cast<std::complex<float>*>(
          &m_calibrate_state.visibilities.Span()(antenna_nr, 0, 0, 0, 0, 0));
  float* weights_ptr =
      &m_calibrate_state.weights.Span()(antenna_nr, 0, 0, 0, 0, 0);
  std::complex<float>* subgrids_ptr =
      m_calibrate_state.subgrids[antenna_nr].Span().data();
  std::complex<float>* phasors_ptr =
      m_calibrate_state.phasors[antenna_nr].Span().data();
  double* hessian_ptr = hessian.data();
  double* gradient_ptr = gradient.data();
  double* residual_ptr = residual.data();

  const size_t max_nr_timesteps =
      m_calibrate_state.max_nr_timesteps[antenna_nr];

  // Run calibration update step
  m_kernels->run_calibrate(
      nr_subgrids, nr_polarizations, grid_size, subgrid_size, image_size,
      w_step, shift_ptr, max_nr_timesteps, nr_channels, nr_terms, nr_stations,
      nr_timeslots, uvw_ptr, wavenumbers_ptr, visibilities_ptr, weights_ptr,
      aterm_ptr, aterm_derivative_ptr, aterm_idx_ptr, metadata_ptr,
      subgrids_ptr, phasors_ptr, hessian_ptr, gradient_ptr, residual_ptr);

  // Performance reporting
  const size_t current_nr_subgrids = nr_subgrids;
  const size_t current_nr_timesteps =
      m_calibrate_state.plans[antenna_nr][0]->get_nr_timesteps();
  const size_t current_nr_visibilities = current_nr_timesteps * nr_channels;
  get_report()->update_total(current_nr_subgrids, current_nr_timesteps,
                             current_nr_visibilities);
}

void CPU::do_calibrate_finish() {
  // Performance reporting
  const size_t nr_antennas = m_calibrate_state.plans.size();
  size_t total_nr_timesteps = 0;
  size_t total_nr_subgrids = 0;
  for (size_t antenna_nr = 0; antenna_nr < nr_antennas; antenna_nr++) {
    total_nr_timesteps +=
        m_calibrate_state.plans[antenna_nr][0]->get_nr_timesteps();
    total_nr_subgrids +=
        m_calibrate_state.plans[antenna_nr][0]->get_nr_subgrids();
  }
  get_report()->print_total(nr_correlations, total_nr_timesteps,
                            total_nr_subgrids);
  get_report()->print_visibilities(auxiliary::name_calibrate);
}

void CPU::do_transform(DomainAtoDomainB direction) {
#if defined(DEBUG)
  std::cout << __func__ << std::endl;
  std::cout << "FFT (direction: " << direction << ")" << std::endl;
#endif

  try {
    const auto& grid = get_final_grid();

    // Constants
    const size_t nr_w_layers = grid.shape(0);
    const size_t nr_correlations = grid.shape(1);
    const size_t grid_size = grid.shape(2);
    assert(grid.shape(3) == grid_size);

    // Performance measurement
    get_report()->initialize(0, 0, grid_size);
    m_kernels->set_report(get_report());

    for (size_t w = 0; w < nr_w_layers; w++) {
      const int sign = (direction == FourierDomainToImageDomain) ? 1 : -1;

      aocommon::xt::Span<std::complex<float>, 3> grid_w =
          aocommon::xt::CreateSpan<std::complex<float>, 3>(
              &get_grid()(w, 0, 0, 0), {nr_correlations, grid_size, grid_size});

      pmt::State states[2];
      states[0] = power_meter_->Read();
      m_kernels->fftshift_grid(grid_w);
      m_kernels->run_fft(grid_size, grid_size, nr_correlations, grid_w.data(),
                         sign);
      m_kernels->fftshift_grid(grid_w);
      states[1] = power_meter_->Read();
      get_report()->update(Report::host, states[0], states[1]);
    }

    // Report performance
    get_report()->print_total(nr_correlations);
    std::clog << std::endl;

  } catch (const std::exception& e) {
    std::cerr << __func__ << " caught exception: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << __func__ << " caught unknown exception" << std::endl;
  }
}  // end transform

void CPU::do_compute_avg_beam(
    const unsigned int nr_antennas, const unsigned int nr_channels,
    const aocommon::xt::Span<UVW<float>, 2>& uvw,
    const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
        baselines,
    const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
    const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
    const aocommon::xt::Span<float, 4>& weights,
    aocommon::xt::Span<std::complex<float>, 4>& average_beam) {
#if defined(DEBUG)
  std::cout << __func__ << std::endl;
#endif

  const size_t nr_aterms = aterm_offsets.size() - 1;
  const size_t nr_baselines = baselines.size();
  assert(uvw.shape(0) == nr_baselines);
  const size_t nr_timesteps = uvw.shape(1);
  const size_t subgrid_size = average_beam.shape(0);
  assert(average_beam.shape(1) == subgrid_size);
  const size_t nr_polarizations = 4;

  get_report()->initialize();
  m_kernels->set_report(get_report());

  auto* baselines_ptr = reinterpret_cast<const Baseline*>(baselines.data());
  auto* aterms_ptr =
      reinterpret_cast<const std::complex<float>*>(aterms.data());

  m_kernels->run_average_beam(
      nr_baselines, nr_antennas, nr_timesteps, nr_channels, nr_aterms,
      subgrid_size, nr_polarizations, uvw.data(), baselines_ptr, aterms_ptr,
      aterm_offsets.data(), weights.data(), average_beam.data());

  get_report()->print_total(nr_correlations);
}  // end compute_avg_beam

}  // namespace cpu
}  // namespace proxy
}  // namespace idg
