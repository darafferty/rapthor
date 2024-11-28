// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <vector>
#include <memory>
#include <climits>

#include <xtensor/xview.hpp>

#include "fftw3.h"

#include "CPU.h"

#ifdef HAVE_LIBDIRAC
#include <Dirac.h>
#undef complex
#endif /* HAVE_LIBDIRAC */

// #define DEBUG_COMPUTE_JOBSIZE

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

namespace {

#ifdef HAVE_LIBDIRAC
struct lbfgs_idgcal_data {
  const std::shared_ptr<kernel::cpu::InstanceCPU> m_kernels;
  const aocommon::xt::Span<double, 4>& phase_basis;
  aocommon::xt::Span<Matrix2x2<std::complex<double>>, 5>& aterm;
  aocommon::xt::Span<Matrix2x2<std::complex<double>>, 5>& aterm_deriv;
  aocommon::xt::Span<double, 4>& phase;
  // gradient is not used in cost calculation
  aocommon::xt::Span<double, 3>& local_gradient;
  // following are fields in (unnamed) struct CPU::m_calibrate_state
  std::vector<std::vector<std::unique_ptr<Plan>>>* plans;
  size_t nr_baselines;
  size_t nr_timesteps;
  size_t nr_channels;
  Tensor<float, 1>* wavenumbers;
  Tensor<std::complex<float>, 6>* visibilities;
  Tensor<float, 6>* weights;
  Tensor<UVW<float>, 3>* uvw;
  Tensor<std::pair<unsigned int, unsigned int>, 2>* baselines;
  std::vector<Tensor<std::complex<float>, 4>>* subgrids;
  std::vector<Tensor<std::complex<float>, 4>>* phasors;
  std::vector<int>* max_nr_timesteps;
  // following are fields in m_cache_state
  size_t nr_polarizations;
  size_t grid_size;
  float image_size;
  float w_step;
  float* shift_ptr;
  // following are from input (python) side
  size_t nr_channel_blocks;
  size_t subgrid_size;
  size_t nr_antennas;
  size_t nr_timeslots;
  size_t nr_terms;
  size_t nr_correlations;
  // Note: some of the above might be duplicate/not needed - cleanup TBD

  lbfgs_idgcal_data(
      std::shared_ptr<kernel::cpu::InstanceCPU> m_kernels_,
      aocommon::xt::Span<double, 4>& phase_basis_,
      aocommon::xt::Span<Matrix2x2<std::complex<double>>, 5>& aterm_,
      aocommon::xt::Span<Matrix2x2<std::complex<double>>, 5>& aterm_deriv_,
      aocommon::xt::Span<double, 4>& phase_,
      aocommon::xt::Span<double, 3>& local_gradient_,
      std::vector<std::vector<std::unique_ptr<Plan>>>* plans_,
      size_t nr_baselines_, size_t nr_timesteps_, size_t nr_channels_,
      Tensor<float, 1>* wavenumbers_,
      Tensor<std::complex<float>, 6>* visibilities_, Tensor<float, 6>* weights_,
      Tensor<UVW<float>, 3>* uvw_,
      Tensor<std::pair<unsigned int, unsigned int>, 2>* baselines_,
      std::vector<Tensor<std::complex<float>, 4>>* subgrids_,
      std::vector<Tensor<std::complex<float>, 4>>* phasors_,
      std::vector<int>* max_nr_timesteps_, size_t nr_polarizations_,
      size_t grid_size_, float image_size_, float w_step_, float* shift_ptr_,
      size_t nr_channel_blocks_, size_t subgrid_size_, size_t nr_antennas_,
      size_t nr_timeslots_, size_t nr_terms_, size_t nr_correlations_)
      : m_kernels(m_kernels_),
        phase_basis(phase_basis_),
        aterm(aterm_),
        aterm_deriv(aterm_deriv_),
        phase(phase_),
        local_gradient(local_gradient_),
        plans(plans_),
        nr_baselines(nr_baselines_),
        nr_timesteps(nr_timesteps_),
        nr_channels(nr_channels_),
        wavenumbers(wavenumbers_),
        visibilities(visibilities_),
        weights(weights_),
        uvw(uvw_),
        baselines(baselines_),
        subgrids(subgrids_),
        phasors(phasors_),
        max_nr_timesteps(max_nr_timesteps_),
        nr_polarizations(nr_polarizations_),
        grid_size(grid_size_),
        image_size(image_size_),
        w_step(w_step_),
        shift_ptr(shift_ptr_),
        nr_channel_blocks(nr_channel_blocks_),
        subgrid_size(subgrid_size_),
        nr_antennas(nr_antennas_),
        nr_timeslots(nr_timeslots_),
        nr_terms(nr_terms_),
        nr_correlations(nr_correlations){};
};
#endif /* HAVE_LIBDIRAC */

void do_calc_phase(const int nr_channel_blocks, const int subgrid_size,
                   const int nr_antennas, const int nr_terms,
                   aocommon::xt::Span<double, 3>& parameters,
                   const aocommon::xt::Span<double, 4>& phase_basis,
                   aocommon::xt::Span<double, 4>& phase) {
  for (size_t i = 0; i < (size_t)nr_channel_blocks; i++) {
    for (size_t j = 0; j < (size_t)nr_antennas; j++) {
      for (size_t k = 0; k < (size_t)subgrid_size; k++) {
#pragma omp parallel for
        for (size_t l = 0; l < (size_t)subgrid_size; l++) {
          double product = 0.0;
#pragma GCC ivdep
          for (size_t m = 0; m < (size_t)nr_terms; m++) {
            // 0 for only using XX correlation of basis (scalar basis)
            product += parameters(i, j, m) * phase_basis(m, k, l, 0);
          }
          phase(i, j, k, l) = product;
        }
      }
    }
  }
}

void do_calc_aterms(
    const int nr_channel_blocks, const int subgrid_size, const int nr_antennas,
    aocommon::xt::Span<double, 4>& phase,
    aocommon::xt::Span<Matrix2x2<std::complex<double>>, 5>& aterm) {
  for (size_t i = 0; i < (size_t)nr_channel_blocks; i++) {
    for (size_t j = 0; j < (size_t)nr_antennas; j++) {
      for (size_t k = 0; k < (size_t)subgrid_size; k++) {
        for (size_t l = 0; l < (size_t)subgrid_size; l++) {
          // 0 for having nr_phase_updates==1
          Matrix2x2<std::complex<double>>& mat = aterm(i, 0, j, k, l);
          double s, c;
          sincos(phase(i, j, k, l), &s, &c);
          // xy and yx are already set to zero
          mat.xx.real(c);
          mat.yy.real(c);
          mat.xx.imag(s);
          mat.yy.imag(s);
        }
      }
    }
  }
}

void do_calc_aterm_derivatives(
    const int nr_channel_blocks, const int subgrid_size, const int antenna_nr,
    const int nr_terms, const aocommon::xt::Span<double, 4>& phase_basis,
    aocommon::xt::Span<Matrix2x2<std::complex<double>>, 5>& aterm,
    aocommon::xt::Span<Matrix2x2<std::complex<double>>, 5>& aterm_deriv) {
  for (size_t i = 0; i < (size_t)nr_channel_blocks; i++) {
    for (size_t k = 0; k < (size_t)subgrid_size; k++) {
      for (size_t l = 0; l < (size_t)subgrid_size; l++) {
        // 0 for having nr_phase_updates==1
        Matrix2x2<std::complex<double>>& phase_mat =
            aterm(i, 0, antenna_nr, k, l);
        for (size_t m = 0; m < (size_t)nr_terms; m++) {
          // 0 for having only one antenna
          Matrix2x2<std::complex<double>>& deriv_mat =
              aterm_deriv(i, 0, m, k, l);
          // scalar basis, to multiply phase_mat with a (scalar x j)
          double basis = phase_basis(m, k, l, 0);
          double xx_re = phase_mat.xx.real() * basis;
          double xx_im = phase_mat.xx.imag() * basis;
          double yy_re = phase_mat.yy.real() * basis;
          double yy_im = phase_mat.yy.imag() * basis;
          deriv_mat.xx.real(-xx_im);
          deriv_mat.xx.imag(xx_re);
          deriv_mat.yy.real(-yy_im);
          deriv_mat.yy.imag(yy_re);
        }
      }
    }
  }
}

#ifdef HAVE_LIBDIRAC
double lbfgs_cost_function(double* unknowns, int n_unknowns, void* extra_data) {
  assert(extra_data);
  const lbfgs_idgcal_data* lt =
      reinterpret_cast<lbfgs_idgcal_data*>(extra_data);
  assert(lt->wavenumbers);
  assert(lt->visibilities);
  assert(lt->weights);
  assert(lt->uvw);
  assert(lt->subgrids);
  assert(lt->phasors);
  assert(lt->max_nr_timesteps);
  assert(lt->shift_ptr);
  if (lt->plans->empty()) {
    throw std::runtime_error("Calibration was not initialized. Can not update");
  }

  const std::array<size_t, 3> parameters_shape{
      static_cast<size_t>(lt->nr_channel_blocks),
      static_cast<size_t>(lt->nr_antennas), static_cast<size_t>(lt->nr_terms)};

  aocommon::xt::Span<double, 3> parameters =
      aocommon::xt::CreateSpan(unknowns, parameters_shape);

  do_calc_phase(lt->nr_channel_blocks, lt->subgrid_size, lt->nr_antennas,
                lt->nr_terms, parameters, lt->phase_basis, lt->phase);
  do_calc_aterms(lt->nr_channel_blocks, lt->subgrid_size, lt->nr_antennas,
                 lt->phase, lt->aterm);
  const std::complex<double>* aterm_ptr =
      reinterpret_cast<const std::complex<double>*>(lt->aterm.data());

  float* wavenumbers_ptr = (*(lt->wavenumbers)).Span().data();
  float* shift_ptr = lt->shift_ptr;

  double total_residual = 0.0;
  double residual;
  double* residual_ptr = &residual;
  for (size_t antenna_nr = 0; antenna_nr < lt->nr_antennas; antenna_nr++) {
    const size_t nr_subgrids = (*(lt->plans))[antenna_nr][0]->get_nr_subgrids();
    const unsigned int* aterm_idx_ptr =
        (*(lt->plans))[antenna_nr][0]->get_aterm_indices_ptr();
    const Metadata* metadata_ptr =
        (*(lt->plans))[antenna_nr][0]->get_metadata_ptr();
    const size_t max_nr_timesteps = (*(lt->max_nr_timesteps))[antenna_nr];
    UVW<float>* uvw_ptr = &(*(lt->uvw)).Span()(antenna_nr, 0, 0);
    std::complex<float>* visibilities_ptr =
        reinterpret_cast<std::complex<float>*>(
            &(*(lt->visibilities)).Span()(antenna_nr, 0, 0, 0, 0, 0));
    float* weights_ptr = &(*(lt->weights)).Span()(antenna_nr, 0, 0, 0, 0, 0);
    std::complex<float>* subgrids_ptr =
        (*(lt->subgrids))[antenna_nr].Span().data();
    std::complex<float>* phasors_ptr =
        (*(lt->phasors))[antenna_nr].Span().data();
    residual = 0.0;
    lt->m_kernels->run_calc_cost(
        nr_subgrids, lt->nr_polarizations, lt->grid_size, lt->subgrid_size,
        lt->image_size, lt->w_step, shift_ptr, max_nr_timesteps,
        lt->nr_channels, lt->nr_terms, lt->nr_antennas, lt->nr_timeslots,
        uvw_ptr, wavenumbers_ptr, visibilities_ptr, weights_ptr, aterm_ptr,
        aterm_idx_ptr, metadata_ptr, subgrids_ptr, phasors_ptr, residual_ptr);
    total_residual += residual;
  }

  return total_residual;
}

void lbfgs_grad_function(double* unknowns, double* gradient, int n_unknowns,
                         void* extra_data) {
  assert(extra_data);
  const lbfgs_idgcal_data* lt =
      reinterpret_cast<lbfgs_idgcal_data*>(extra_data);
  assert(lt->wavenumbers);
  assert(lt->visibilities);
  assert(lt->weights);
  assert(lt->uvw);
  assert(lt->subgrids);
  assert(lt->phasors);
  assert(lt->max_nr_timesteps);
  assert(lt->shift_ptr);
  if (lt->plans->empty()) {
    throw std::runtime_error("Calibration was not initialized. Can not update");
  }

  const std::array<size_t, 3> parameters_shape{
      static_cast<size_t>(lt->nr_channel_blocks),
      static_cast<size_t>(lt->nr_antennas), static_cast<size_t>(lt->nr_terms)};

  aocommon::xt::Span<double, 3> parameters =
      aocommon::xt::CreateSpan(unknowns, parameters_shape);
  aocommon::xt::Span<double, 3> global_gradient =
      aocommon::xt::CreateSpan(gradient, parameters_shape);

  do_calc_phase(lt->nr_channel_blocks, lt->subgrid_size, lt->nr_antennas,
                lt->nr_terms, parameters, lt->phase_basis, lt->phase);
  do_calc_aterms(lt->nr_channel_blocks, lt->subgrid_size, lt->nr_antennas,
                 lt->phase, lt->aterm);
  const std::complex<double>* aterm_ptr =
      reinterpret_cast<const std::complex<double>*>(lt->aterm.data());
  const std::complex<double>* aterm_derivative_ptr =
      reinterpret_cast<const std::complex<double>*>(lt->aterm_deriv.data());

  float* wavenumbers_ptr = (*(lt->wavenumbers)).Span().data();
  float* shift_ptr = lt->shift_ptr;

  for (size_t antenna_nr = 0; antenna_nr < lt->nr_antennas; antenna_nr++) {
    const size_t nr_subgrids = (*(lt->plans))[antenna_nr][0]->get_nr_subgrids();
    const unsigned int* aterm_idx_ptr =
        (*(lt->plans))[antenna_nr][0]->get_aterm_indices_ptr();
    const Metadata* metadata_ptr =
        (*(lt->plans))[antenna_nr][0]->get_metadata_ptr();
    const size_t max_nr_timesteps = (*(lt->max_nr_timesteps))[antenna_nr];
    UVW<float>* uvw_ptr = &(*(lt->uvw)).Span()(antenna_nr, 0, 0);
    std::complex<float>* visibilities_ptr =
        reinterpret_cast<std::complex<float>*>(
            &(*(lt->visibilities)).Span()(antenna_nr, 0, 0, 0, 0, 0));
    float* weights_ptr = &(*(lt->weights)).Span()(antenna_nr, 0, 0, 0, 0, 0);
    std::complex<float>* subgrids_ptr =
        (*(lt->subgrids))[antenna_nr].Span().data();
    std::complex<float>* phasors_ptr =
        (*(lt->phasors))[antenna_nr].Span().data();

    // Update aterm_deriv for this antenna
    memset(static_cast<void*>(lt->aterm_deriv.data()), 0,
           sizeof(std::complex<double>) * lt->nr_channel_blocks * lt->nr_terms *
               lt->subgrid_size * lt->subgrid_size * 4);

    do_calc_aterm_derivatives(lt->nr_channel_blocks, lt->subgrid_size,
                              antenna_nr, lt->nr_terms, lt->phase_basis,
                              lt->aterm, lt->aterm_deriv);

    double* gradient_ptr = lt->local_gradient.data();
    lt->m_kernels->run_calc_gradient(
        nr_subgrids, lt->nr_polarizations, lt->grid_size, lt->subgrid_size,
        lt->image_size, lt->w_step, shift_ptr, max_nr_timesteps,
        lt->nr_channels, lt->nr_terms, lt->nr_antennas, lt->nr_timeslots,
        uvw_ptr, wavenumbers_ptr, visibilities_ptr, weights_ptr, aterm_ptr,
        aterm_derivative_ptr, aterm_idx_ptr, metadata_ptr, subgrids_ptr,
        phasors_ptr, gradient_ptr);

    for (size_t i = 0; i < global_gradient.shape(0); i++) {
#pragma omp parallel for
      for (size_t j = 0; j < global_gradient.shape(2); j++) {
        global_gradient(i, antenna_nr, j) = lt->local_gradient(i, 0, j);
      }
    }
  }
}
#endif /* HAVE_LIBDIRAC */

}  // namespace

// Following is obsolete, should be removed (after testing)
void CPU::do_calc_cost(const int nr_channel_blocks, const int subgrid_size,
                       const int nr_antennas, const int nr_timeslots,
                       const int nr_terms, const int nr_correlations,
                       aocommon::xt::Span<double, 3>& parameters,
                       aocommon::xt::Span<double, 4>& phase_basis,
                       aocommon::xt::Span<double, 1>& residual) {
  if (m_calibrate_state.plans.empty()) {
    throw std::runtime_error("Calibration was not initialized. Can not update");
  }

  const int ref_antenna_nr = 0;

  /*
   parameters: channel_blocks x antennas x nr_terms(coeffs), where nr_terms:
   phase_updates x poly_coeffs

   phase_basis: nr_terms(coeffs) x (grid x grid) x corr(=4), assumed to be
   post-multiplied by [1,0; 0,1], hence will be assumed same for all 4
   correlations

   aterms (final shape): channel_blocks x phase_updates(=1) x  stations x (grid
   x grid) x corr(=4) - last dim absorbed in mat2x2 Note: poly_coeffs is
   collapsed in aterms because of dot product, parameters x basis =(collapse
   into 1)

   aterms_derivative (per station, final shape): channel_blocks x
   stations(=1) x poly_coeffs x  (grid x grid) x corr(=4) - last dim absorbed in
   mat2x2
  */

  // Extract metadata
  const size_t nr_channels = m_calibrate_state.wavenumbers.Span().size();
  const size_t nr_polarizations = get_grid().shape(1);
  const size_t grid_size = get_grid().shape(2);
  assert(get_grid().shape(3) == grid_size);
  const float image_size = grid_size * m_cache_state.cell_size;
  const float w_step = m_cache_state.w_step;
  // Remainder of metadata provided by input arguments

  // Initialize performance measurement
  get_report()->initialize(nr_channels, subgrid_size, 0, nr_terms);

  const size_t nr_phase_updates = 1;
  // Create aterms (per station) and aterms_deriv
  aocommon::xt::Span<Matrix2x2<std::complex<double>>, 5> aterm =
      Proxy::allocate_span<Matrix2x2<std::complex<double>>, 5>(
          {(size_t)nr_channel_blocks, (size_t)nr_phase_updates,
           (size_t)nr_antennas, (size_t)subgrid_size, (size_t)subgrid_size});
  const std::complex<double>* aterm_ptr =
      reinterpret_cast<const std::complex<double>*>(aterm.data());
  memset(static_cast<void*>(aterm.data()), 0,
         sizeof(std::complex<double>) * nr_channel_blocks * nr_phase_updates *
             nr_antennas * subgrid_size * subgrid_size * 4);

  // product of parameters[nr_channel_blocks,nr_antennas,0:nr_terms-1] x
  // basis[0:nr_terms-1,grid,grid,corr] ->
  // [nr_channel_blocks,nr_ant,1(nr_phase_updates),grid,grid,corr] ->
  // [nr_channel_blocks,1,nr_ant,grid,grid,corr] == aterms

  // create storage for phase : chan x ant x grid x grid
  // dropping n_phase_updates(=1) and corr(phase is same for all 4 corr)
  aocommon::xt::Span<double, 4> phase = Proxy::allocate_span<double, 4>(
      {(size_t)nr_channel_blocks, (size_t)nr_antennas, (size_t)subgrid_size,
       (size_t)subgrid_size});
  do_calc_phase(nr_channel_blocks, subgrid_size, nr_antennas, nr_terms,
                parameters, phase_basis, phase);

  do_calc_aterms(nr_channel_blocks, subgrid_size, nr_antennas, phase, aterm);

  // Data pointers common to all antennas
  float* shift_ptr = m_cache_state.shift.data();
  float* wavenumbers_ptr = m_calibrate_state.wavenumbers.Span().data();
  double* residual_ptr = residual.data();

  double total_residual = 0;
  for (size_t antenna_nr = 0; antenna_nr < (size_t)nr_antennas; antenna_nr++) {
    const size_t nr_subgrids =
        m_calibrate_state.plans[antenna_nr][0]->get_nr_subgrids();
    const unsigned int* aterm_idx_ptr =
        m_calibrate_state.plans[antenna_nr][0]->get_aterm_indices_ptr();
    const Metadata* metadata_ptr =
        m_calibrate_state.plans[antenna_nr][0]->get_metadata_ptr();
    const size_t max_nr_timesteps =
        m_calibrate_state.max_nr_timesteps[antenna_nr];
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

    residual[0] = 0.0;
    m_kernels->run_calc_cost(
        nr_subgrids, nr_polarizations, grid_size, subgrid_size, image_size,
        w_step, shift_ptr, max_nr_timesteps, nr_channels, nr_terms, nr_antennas,
        nr_timeslots, uvw_ptr, wavenumbers_ptr, visibilities_ptr, weights_ptr,
        aterm_ptr, aterm_idx_ptr, metadata_ptr, subgrids_ptr, phasors_ptr,
        residual_ptr);
    total_residual += residual[0];
  }

  residual[0] = total_residual;
  // Performance reporting
  const size_t current_nr_subgrids =
      m_calibrate_state.plans[ref_antenna_nr][0]->get_nr_subgrids();
  const size_t current_nr_timesteps =
      m_calibrate_state.plans[ref_antenna_nr][0]->get_nr_timesteps();
  const size_t current_nr_visibilities = current_nr_timesteps * nr_channels;
  get_report()->update_total(current_nr_subgrids, current_nr_timesteps,
                             current_nr_visibilities);
}

void CPU::do_lbfgs_fit(const int nr_channel_blocks, const int subgrid_size,
                       const int nr_antennas, const int nr_timeslots,
                       const int nr_terms, const int nr_correlations,
                       const int lbfgs_max_iterations,
                       const int lbfgs_history_size,
                       aocommon::xt::Span<double, 3>& parameters,
                       aocommon::xt::Span<double, 3>& parameters_lower_bound,
                       aocommon::xt::Span<double, 3>& parameters_upper_bound,
                       aocommon::xt::Span<double, 4>& phase_basis,
                       aocommon::xt::Span<double, 1>& residual) {
#ifdef HAVE_LIBDIRAC
  if (m_calibrate_state.plans.empty()) {
    throw std::runtime_error("Calibration was not initialized. Can not update");
  }

  const int ref_antenna_nr = 0;

  /*
   parameters: channel_blocks x antennas x nr_terms(coeffs), where nr_terms:
   phase_updates x poly_coeffs

   phase_basis: nr_terms(coeffs) x (grid x grid) x corr(=4), assumed to be
   post-multiplied by [1,0; 0,1], hence will be assumed same for all 4
   correlations

   aterms (final shape): channel_blocks x phase_updates(=1) x  stations x (grid
   x grid) x corr(=4) - last dim absorbed in mat2x2 Note: poly_coeffs is
   collapsed in aterms because of dot product, parameters x basis =(collapse
   into 1)

   aterms_derivative (per station, final shape): channel_blocks x
   stations(=1) x poly_coeffs x  (grid x grid) x corr(=4) - last dim absorbed in
   mat2x2
  */

  // Extract metadata
  const size_t nr_channels = m_calibrate_state.wavenumbers.Span().size();
  const size_t nr_polarizations = get_grid().shape(1);
  const size_t grid_size = get_grid().shape(2);
  assert(get_grid().shape(3) == grid_size);
  const float image_size = grid_size * m_cache_state.cell_size;
  const float w_step = m_cache_state.w_step;
  // Remainder of metadata provided by input arguments

  // Initialize performance measurement
  get_report()->initialize(nr_channels, subgrid_size, 0, nr_terms);

  const size_t nr_phase_updates = 1;
  // Create aterms (per station) and aterms_deriv
  aocommon::xt::Span<Matrix2x2<std::complex<double>>, 5> aterm =
      Proxy::allocate_span<Matrix2x2<std::complex<double>>, 5>(
          {(size_t)nr_channel_blocks, (size_t)nr_phase_updates,
           (size_t)nr_antennas, (size_t)subgrid_size, (size_t)subgrid_size});
  const std::complex<double>* aterm_ptr =
      reinterpret_cast<const std::complex<double>*>(aterm.data());
  memset(static_cast<void*>(aterm.data()), 0,
         sizeof(std::complex<double>) * nr_channel_blocks * nr_phase_updates *
             nr_antennas * subgrid_size * subgrid_size * 4);

  // product of parameters[nr_channel_blocks,nr_antennas,0:nr_terms-1] x
  // basis[0:nr_terms-1,grid,grid,corr] ->
  // [nr_channel_blocks,nr_ant,1(nr_phase_updates),grid,grid,corr] ->
  // [nr_channel_blocks,1,nr_ant,grid,grid,corr] == aterms

  // create storage for phase : chan x ant x grid x grid
  // dropping n_phase_updates(=1) and corr(phase is same for all 4 corr)
  aocommon::xt::Span<double, 4> phase = Proxy::allocate_span<double, 4>(
      {(size_t)nr_channel_blocks, (size_t)nr_antennas, (size_t)subgrid_size,
       (size_t)subgrid_size});

  // 1 for one antenna
  aocommon::xt::Span<Matrix2x2<std::complex<double>>, 5> aterm_deriv =
      Proxy::allocate_span<Matrix2x2<std::complex<double>>, 5>(
          {(size_t)nr_channel_blocks, 1, (size_t)nr_terms, (size_t)subgrid_size,
           (size_t)subgrid_size});

  // gradient storage is not used
  aocommon::xt::Span<double, 3> local_gradient =
      Proxy::allocate_span<double, 3>(
          {(size_t)nr_channel_blocks, (size_t)1, (size_t)nr_terms});

  lbfgs_idgcal_data lbfgs_dat(
      m_kernels, phase_basis, aterm, aterm_deriv, phase, local_gradient,
      &m_calibrate_state.plans, m_calibrate_state.nr_baselines,
      m_calibrate_state.nr_timesteps, m_calibrate_state.nr_channels,
      &m_calibrate_state.wavenumbers, &m_calibrate_state.visibilities,
      &m_calibrate_state.weights, &m_calibrate_state.uvw,
      &m_calibrate_state.baselines, &m_calibrate_state.subgrids,
      &m_calibrate_state.phasors, &m_calibrate_state.max_nr_timesteps,
      nr_polarizations, grid_size, image_size, w_step,
      m_cache_state.shift.data(), (size_t)nr_channel_blocks,
      (size_t)subgrid_size, (size_t)nr_antennas, (size_t)nr_timeslots,
      (size_t)nr_terms, (size_t)nr_correlations);

  int n_solutions = parameters.size();
  lbfgsb_fit(lbfgs_cost_function, lbfgs_grad_function, parameters.data(),
             parameters_lower_bound.data(), parameters_upper_bound.data(),
             n_solutions, lbfgs_max_iterations, lbfgs_history_size,
             (void*)&lbfgs_dat, nullptr);

  // calculate residual
  residual[0] =
      lbfgs_cost_function(parameters.data(), n_solutions, (void*)&lbfgs_dat);
  // Performance reporting
  const size_t current_nr_subgrids =
      m_calibrate_state.plans[ref_antenna_nr][0]->get_nr_subgrids();
  const size_t current_nr_timesteps =
      m_calibrate_state.plans[ref_antenna_nr][0]->get_nr_timesteps();
  const size_t current_nr_visibilities = current_nr_timesteps * nr_channels;
  get_report()->update_total(current_nr_subgrids, current_nr_timesteps,
                             current_nr_visibilities);

#endif /* HAVE_LIBDIRAC */
}

// Following is obsolete, should be removed (after testing)
void CPU::do_calc_gradient(const int nr_channel_blocks, const int subgrid_size,
                           const int nr_antennas, const int nr_timeslots,
                           const int nr_terms, const int nr_correlations,
                           aocommon::xt::Span<double, 3>& parameters,
                           aocommon::xt::Span<double, 4>& phase_basis,
                           aocommon::xt::Span<double, 3>& gradient) {
  if (m_calibrate_state.plans.empty()) {
    throw std::runtime_error("Calibration was not initialized. Can not update");
  }

  const int ref_antenna_nr = 0;

  // Extract metadata
  const size_t nr_channels = m_calibrate_state.wavenumbers.Span().size();
  const size_t nr_polarizations = get_grid().shape(1);
  const size_t grid_size = get_grid().shape(2);
  assert(get_grid().shape(3) == grid_size);
  const float image_size = grid_size * m_cache_state.cell_size;
  const float w_step = m_cache_state.w_step;
  // Remainder of metadata provided by input arguments

  // Initialize performance measurement
  get_report()->initialize(nr_channels, subgrid_size, 0, nr_terms);

  const size_t nr_phase_updates = 1;  // equal to nr_timeslots ??
  // Create aterms (per station) and aterms_deriv
  aocommon::xt::Span<Matrix2x2<std::complex<double>>, 5> aterm =
      Proxy::allocate_span<Matrix2x2<std::complex<double>>, 5>(
          {(size_t)nr_channel_blocks, (size_t)nr_phase_updates,
           (size_t)nr_antennas, (size_t)subgrid_size, (size_t)subgrid_size});
  // 1 for one antenna
  aocommon::xt::Span<Matrix2x2<std::complex<double>>, 5> aterm_deriv =
      Proxy::allocate_span<Matrix2x2<std::complex<double>>, 5>(
          {(size_t)nr_channel_blocks, 1, (size_t)nr_terms, (size_t)subgrid_size,
           (size_t)subgrid_size});
  const std::complex<double>* aterm_ptr =
      reinterpret_cast<const std::complex<double>*>(aterm.data());
  const std::complex<double>* aterm_derivative_ptr =
      reinterpret_cast<const std::complex<double>*>(aterm_deriv.data());
  memset(static_cast<void*>(aterm.data()), 0,
         sizeof(std::complex<double>) * nr_channel_blocks * nr_phase_updates *
             nr_antennas * subgrid_size * subgrid_size * 4);

  // product of parameters[nr_channel_blocks,nr_antennas,0:nr_terms-1] x
  // basis[0:nr_terms-1,grid,grid,corr] ->
  // [nr_channel_blocks,nr_ant,1(nr_phase_updates),grid,grid,corr] ->
  // [nr_channel_blocks,1,nr_ant,grid,grid,corr] == aterms

  // create storage for phase : chan x ant x grid x grid
  // dropping n_phase_updates(=1) and corr(phase is same for all 4 corr)
  aocommon::xt::Span<double, 4> phase = Proxy::allocate_span<double, 4>(
      {(size_t)nr_channel_blocks, (size_t)nr_antennas, (size_t)subgrid_size,
       (size_t)subgrid_size});
  do_calc_phase(nr_channel_blocks, subgrid_size, nr_antennas, nr_terms,
                parameters, phase_basis, phase);

  do_calc_aterms(nr_channel_blocks, subgrid_size, nr_antennas, phase, aterm);

  // Data pointers common to all antennas
  float* shift_ptr = m_cache_state.shift.data();
  float* wavenumbers_ptr = m_calibrate_state.wavenumbers.Span().data();
  // Gradient for accumulation for one antenna
  aocommon::xt::Span<double, 3> local_gradient =
      Proxy::allocate_span<double, 3>(
          {(size_t)nr_channel_blocks, (size_t)1, (size_t)nr_terms});
  /* const std::array<size_t, 3> param_shape{
      static_cast<size_t>(nr_channel_blocks), static_cast<size_t>(nr_antennas),
      static_cast<size_t>(nr_terms)};

      const std::array<size_t, 3> gradient_shape{
      static_cast<size_t>(nr_channel_blocks), static_cast<size_t>(nr_timeslots),
      static_cast<size_t>(nr_terms)};
      */

  for (size_t antenna_nr = 0; antenna_nr < (size_t)nr_antennas; antenna_nr++) {
    const size_t nr_subgrids =
        m_calibrate_state.plans[antenna_nr][0]->get_nr_subgrids();
    const unsigned int* aterm_idx_ptr =
        m_calibrate_state.plans[antenna_nr][0]->get_aterm_indices_ptr();
    const Metadata* metadata_ptr =
        m_calibrate_state.plans[antenna_nr][0]->get_metadata_ptr();
    const size_t max_nr_timesteps =
        m_calibrate_state.max_nr_timesteps[antenna_nr];
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

    // Update aterm_deriv for this antenna
    memset(static_cast<void*>(aterm_deriv.data()), 0,
           sizeof(std::complex<double>) * nr_channel_blocks * nr_terms *
               subgrid_size * subgrid_size * 4);

    do_calc_aterm_derivatives(nr_channel_blocks, subgrid_size, antenna_nr,
                              nr_terms, phase_basis, aterm, aterm_deriv);

    double* gradient_ptr = local_gradient.data();
    m_kernels->run_calc_gradient(
        nr_subgrids, nr_polarizations, grid_size, subgrid_size, image_size,
        w_step, shift_ptr, max_nr_timesteps, nr_channels, nr_terms, nr_antennas,
        nr_timeslots, uvw_ptr, wavenumbers_ptr, visibilities_ptr, weights_ptr,
        aterm_ptr, aterm_derivative_ptr, aterm_idx_ptr, metadata_ptr,
        subgrids_ptr, phasors_ptr, gradient_ptr);

    for (size_t i = 0; i < gradient.shape(0); i++) {
      for (size_t j = 0; j < gradient.shape(2); j++) {
        gradient(i, antenna_nr, j) = local_gradient(i, 0, j);
      }
    }
  }

  // Performance reporting
  const size_t current_nr_subgrids =
      m_calibrate_state.plans[ref_antenna_nr][0]->get_nr_subgrids();
  const size_t current_nr_timesteps =
      m_calibrate_state.plans[ref_antenna_nr][0]->get_nr_timesteps();
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
