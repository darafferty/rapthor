// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <vector>
#include <memory>
#include <climits>

#include "fftw3.h"

#include "CPU.h"

//#define DEBUG_COMPUTE_JOBSIZE

using namespace idg::kernel;
using namespace powersensor;

namespace idg {
namespace proxy {
namespace cpu {

// Constructor
CPU::CPU() {
#if defined(DEBUG)
  std::cout << "CPU::" << __func__ << std::endl;
#endif

  m_powersensor.reset(get_power_sensor(sensor_host));
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
    const int kernel_size, const Array1D<float> &frequencies,
    const Array2D<UVW<float>> &uvw,
    const Array1D<std::pair<unsigned int, unsigned int>> &baselines,
    const Array1D<unsigned int> &aterms_offsets, Plan::Options options) {
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

void CPU::init_cache(int subgrid_size, float cell_size, float w_step,
                     const Array1D<float> &shift) {
  Proxy::init_cache(subgrid_size, cell_size, w_step, shift);
  const int nr_polarizations = m_grid->get_z_dim();
  const size_t grid_size = m_grid->get_x_dim();
  const int nr_wtiles =
      m_kernels->init_wtiles(nr_polarizations, grid_size, subgrid_size);
  m_wtiles = WTiles(nr_wtiles, kernel::cpu::InstanceCPU::kWTileSize);
}

std::shared_ptr<Grid> CPU::get_final_grid() {
  // flush all pending Wtiles
  WTileUpdateInfo wtile_flush_info = m_wtiles.clear();
  if (wtile_flush_info.wtile_ids.size()) {
    auto nr_polarizations = m_grid->get_z_dim();
    auto grid_size = m_grid->get_x_dim();
    auto image_size = grid_size * m_cache_state.cell_size;
    auto subgrid_size = m_cache_state.subgrid_size;
    auto w_step = m_cache_state.w_step;
    auto &shift = m_cache_state.shift;
    State states[2];
    m_report->initialize(0, subgrid_size, grid_size);
    states[0] = m_powersensor->read();
    m_kernels->run_adder_tiles_to_grid(
        nr_polarizations, grid_size, subgrid_size, image_size, w_step,
        shift.data(), wtile_flush_info.wtile_ids.size(),
        wtile_flush_info.wtile_ids.data(),
        wtile_flush_info.wtile_coordinates.data(), m_grid->data());
    states[1] = m_powersensor->read();
    m_report->update(Report::wtiling_forward, states[0], states[1]);
    m_report->print_total(nr_correlations);
  }
  return m_grid;
}

unsigned int CPU::compute_jobsize(const Plan &plan,
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

  return jobsize;
}

/*
    High level routines
*/
void CPU::do_gridding(
    const Plan &plan, const Array1D<float> &frequencies,
    const Array4D<std::complex<float>> &visibilities,
    const Array2D<UVW<float>> &uvw,
    const Array1D<std::pair<unsigned int, unsigned int>> &baselines,
    const Array4D<Matrix2x2<std::complex<float>>> &aterms,
    const Array1D<unsigned int> &aterms_offsets,
    const Array2D<float> &spheroidal) {
#if defined(DEBUG)
  std::cout << __func__ << std::endl;
#endif

  m_kernels->set_report(m_report);

  Array1D<float> wavenumbers = compute_wavenumbers(frequencies);

  // Arguments
  auto nr_baselines = visibilities.get_w_dim();
  auto nr_timesteps = visibilities.get_z_dim();
  auto nr_channels = visibilities.get_y_dim();
  auto nr_correlations = visibilities.get_x_dim();
  auto nr_polarizations = m_grid->get_z_dim();
  auto grid_size = m_grid->get_x_dim();
  auto subgrid_size = plan.get_subgrid_size();
  auto &shift = plan.get_shift();
  auto w_step = plan.get_w_step();
  auto image_size = plan.get_cell_size() * grid_size;
  auto nr_stations = aterms.get_z_dim();

  WTileUpdateSet wtile_flush_set = plan.get_wtile_flush_set();

  try {
    auto jobsize =
        compute_jobsize(plan, nr_timesteps, nr_channels, nr_correlations,
                        nr_polarizations, subgrid_size);

    // Allocate memory for subgrids
    int max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);
    Array4D<std::complex<float>> subgrids(max_nr_subgrids, nr_correlations,
                                          subgrid_size, subgrid_size);

    // Performance measurement
    m_report->initialize(nr_channels, subgrid_size, grid_size);
    State states[2];
    states[0] = m_powersensor->read();

    // Start gridder
    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
      unsigned int first_bl, last_bl, current_nr_baselines;
      plan.initialize_job(nr_baselines, jobsize, bl, &first_bl, &last_bl,
                          &current_nr_baselines);
      if (current_nr_baselines == 0) continue;

      // Initialize iteration
      auto current_nr_subgrids =
          plan.get_nr_subgrids(first_bl, current_nr_baselines);
      const float *shift_ptr = shift.data();
      auto *wavenumbers_ptr = wavenumbers.data();
      auto *spheroidal_ptr = spheroidal.data();
      auto *aterm_ptr = reinterpret_cast<std::complex<float> *>(aterms.data());
      auto *aterm_idx_ptr = plan.get_aterm_indices_ptr();
      auto *avg_aterm_ptr = m_avg_aterm_correction.size()
                                ? m_avg_aterm_correction.data()
                                : nullptr;
      auto *metadata_ptr = plan.get_metadata_ptr(first_bl);
      auto *uvw_ptr = uvw.data(0, 0);
      auto *visibilities_ptr =
          reinterpret_cast<std::complex<float> *>(visibilities.data(0, 0, 0));
      auto *subgrids_ptr = subgrids.data(0, 0, 0, 0);
      std::complex<float> *grid_ptr = m_grid->data();

      // Gridder kernel
      m_kernels->run_gridder(current_nr_subgrids, nr_polarizations, grid_size,
                             subgrid_size, image_size, w_step, shift_ptr,
                             nr_channels, nr_correlations, nr_stations, uvw_ptr,
                             wavenumbers_ptr, visibilities_ptr, spheroidal_ptr,
                             aterm_ptr, aterm_idx_ptr, avg_aterm_ptr,
                             metadata_ptr, subgrids_ptr);

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
      m_report->print(nr_correlations, current_nr_timesteps,
                      current_nr_subgrids);
    }  // end for bl

    states[1] = m_powersensor->read();
    m_report->update(Report::host, states[0], states[1]);

    // Performance report
    auto total_nr_subgrids = plan.get_nr_subgrids();
    auto total_nr_timesteps = plan.get_nr_timesteps();
    m_report->print_total(nr_correlations, total_nr_timesteps,
                          total_nr_subgrids);
    auto total_nr_visibilities = plan.get_nr_visibilities();
    m_report->print_visibilities(auxiliary::name_gridding,
                                 total_nr_visibilities);

  } catch (const std::invalid_argument &e) {
    std::cerr << __func__ << ": invalid argument: " << e.what() << std::endl;
    exit(1);
  } catch (const std::exception &e) {
    std::cerr << __func__ << ": caught exception: " << e.what() << std::endl;
    exit(2);
  } catch (...) {
    std::cerr << __func__ << ": caught unknown exception" << std::endl;
    exit(3);
  }
}  // end gridding

void CPU::do_degridding(
    const Plan &plan, const Array1D<float> &frequencies,
    Array4D<std::complex<float>> &visibilities, const Array2D<UVW<float>> &uvw,
    const Array1D<std::pair<unsigned int, unsigned int>> &baselines,
    const Array4D<Matrix2x2<std::complex<float>>> &aterms,
    const Array1D<unsigned int> &aterms_offsets,
    const Array2D<float> &spheroidal) {
#if defined(DEBUG)
  std::cout << __func__ << std::endl;
#endif

  m_kernels->set_report(m_report);

  Array1D<float> wavenumbers = compute_wavenumbers(frequencies);

  // Arguments
  auto nr_baselines = visibilities.get_w_dim();
  auto nr_timesteps = visibilities.get_z_dim();
  auto nr_channels = visibilities.get_y_dim();
  auto nr_correlations = visibilities.get_x_dim();
  auto nr_polarizations = m_grid->get_z_dim();
  auto grid_size = m_grid->get_x_dim();
  auto image_size = plan.get_cell_size() * grid_size;
  auto subgrid_size = plan.get_subgrid_size();
  auto w_step = plan.get_w_step();
  auto nr_stations = aterms.get_z_dim();
  auto &shift = plan.get_shift();

  WTileUpdateSet wtile_initialize_set = plan.get_wtile_initialize_set();

  try {
    auto jobsize =
        compute_jobsize(plan, nr_timesteps, nr_channels, nr_correlations,
                        nr_polarizations, subgrid_size);

    // Allocate memory for subgrids
    int max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);
    Array4D<std::complex<float>> subgrids(max_nr_subgrids, nr_correlations,
                                          subgrid_size, subgrid_size);

    // Performance measurement
    m_report->initialize(nr_channels, subgrid_size, grid_size);
    State states[2];
    states[0] = m_powersensor->read();

    // Run subroutines
    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
      unsigned int first_bl, last_bl, current_nr_baselines;
      plan.initialize_job(nr_baselines, jobsize, bl, &first_bl, &last_bl,
                          &current_nr_baselines);
      if (current_nr_baselines == 0) continue;

      // Initialize iteration
      auto current_nr_subgrids =
          plan.get_nr_subgrids(first_bl, current_nr_baselines);
      const float *shift_ptr = shift.data();
      auto *wavenumbers_ptr = wavenumbers.data();
      auto *spheroidal_ptr = spheroidal.data();
      auto *aterm_ptr = reinterpret_cast<std::complex<float> *>(aterms.data());
      auto *aterm_idx_ptr = plan.get_aterm_indices_ptr();
      auto *metadata_ptr = plan.get_metadata_ptr(first_bl);
      auto *uvw_ptr = uvw.data(0, 0);
      auto *visibilities_ptr =
          reinterpret_cast<std::complex<float> *>(visibilities.data(0, 0, 0));
      auto *subgrids_ptr = subgrids.data(0, 0, 0, 0);
      auto *grid_ptr = m_grid->data();

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
          nr_stations, uvw_ptr, wavenumbers_ptr, visibilities_ptr,
          spheroidal_ptr, aterm_ptr, aterm_idx_ptr, metadata_ptr, subgrids_ptr);

      // Performance reporting
      auto current_nr_timesteps =
          plan.get_nr_timesteps(first_bl, current_nr_baselines);
      m_report->print(nr_correlations, current_nr_timesteps,
                      current_nr_subgrids);
    }  // end for bl

    states[1] = m_powersensor->read();
    m_report->update(Report::host, states[0], states[1]);

    // Report performance
    auto total_nr_subgrids = plan.get_nr_subgrids();
    auto total_nr_timesteps = plan.get_nr_timesteps();
    m_report->print_total(nr_correlations, total_nr_timesteps,
                          total_nr_subgrids);
    auto total_nr_visibilities = plan.get_nr_visibilities();
    m_report->print_visibilities(auxiliary::name_degridding,
                                 total_nr_visibilities);

  } catch (const std::invalid_argument &e) {
    std::cerr << __func__ << ": invalid argument: " << e.what() << std::endl;
    exit(1);
  } catch (const std::exception &e) {
    std::cerr << __func__ << ": caught exception: " << e.what() << std::endl;
    exit(2);
  } catch (...) {
    std::cerr << __func__ << ": caught unknown exception" << std::endl;
    exit(3);
  }
}  // end degridding

void CPU::do_calibrate_init(
    std::vector<std::vector<std::unique_ptr<Plan>>> &&plans,
    const Array2D<float> &frequencies,
    Array6D<std::complex<float>> &&visibilities, Array6D<float> &&weights,
    Array3D<UVW<float>> &&uvw,
    Array2D<std::pair<unsigned int, unsigned int>> &&baselines,
    const Array2D<float> &taper) {
  Array1D<float> wavenumbers = compute_wavenumbers(frequencies);

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
  auto nr_channels = visibilities.get_b_dim();
  auto nr_correlations = visibilities.get_a_dim();
  assert(nr_correlations == 4);

  if (nr_channel_blocks > 1) {
    throw std::runtime_error(
        "nr_channel_blocks>1 in calibration is not supported by CPU Proxy.");
  }

  // Allocate subgrids for all antennas
  std::vector<Array4D<std::complex<float>>> subgrids;
  subgrids.reserve(nr_antennas);

  // Allocate phasors for all antennas
  std::vector<Array4D<std::complex<float>>> phasors;
  phasors.reserve(nr_antennas);

  std::vector<int> max_nr_timesteps;
  max_nr_timesteps.reserve(nr_antennas);

  // Start performance measurement
  m_report->initialize();
  powersensor::State states[2];
  states[0] = m_powersensor->read();

  // Create subgrids for every antenna
  for (unsigned int antenna_nr = 0; antenna_nr < nr_antennas; antenna_nr++) {
    // Allocate subgrids for current antenna
    int nr_subgrids = plans[antenna_nr][0]->get_nr_subgrids();
    Array4D<std::complex<float>> subgrids_(nr_subgrids, nr_correlations,
                                           m_cache_state.subgrid_size,
                                           m_cache_state.subgrid_size);

    WTileUpdateSet wtile_initialize_set =
        plans[antenna_nr][0]->get_wtile_initialize_set();

    // Get data pointers
    auto *shift_ptr = m_cache_state.shift.data();
    auto *metadata_ptr = plans[antenna_nr][0]->get_metadata_ptr();
    auto *subgrids_ptr = subgrids_.data();
    std::complex<float> *grid_ptr = m_grid->data();

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

    // Apply spheroidal
    for (int i = 0; i < nr_subgrids; i++) {
      for (unsigned int pol = 0; pol < nr_polarizations; pol++) {
        for (int j = 0; j < subgrid_size; j++) {
          for (int k = 0; k < subgrid_size; k++) {
            int y = (j + (subgrid_size / 2)) % subgrid_size;
            int x = (k + (subgrid_size / 2)) % subgrid_size;
            subgrids_(i, pol, y, x) *= taper(j, k);
          }
        }
      }
    }

    // Store subgrids for current antenna
    subgrids.push_back(std::move(subgrids_));

    // Get max number of timesteps for any subgrid
    auto max_nr_timesteps_ =
        plans[antenna_nr][0]->get_max_nr_timesteps_subgrid();
    max_nr_timesteps.push_back(max_nr_timesteps_);

    // Allocate phasors for current antenna
    Array4D<std::complex<float>> phasors_(nr_subgrids * max_nr_timesteps_,
                                          nr_channels, subgrid_size,
                                          subgrid_size);

    // Compute phasors
    m_kernels->run_calibrate_phasor(
        nr_subgrids, grid_size, subgrid_size, image_size, w_step, shift_ptr,
        max_nr_timesteps_, nr_channels, uvw.data(antenna_nr),
        wavenumbers.data(), metadata_ptr, phasors_.data());

    // Store phasors for current antenna
    phasors.push_back(std::move(phasors_));
  }  // end for antennas

  // End performance measurement
  states[1] = m_powersensor->read();
  m_report->update<Report::ID::host>(states[0], states[1]);
  m_report->print_total(nr_correlations);

  // Set calibration state member variables
  m_calibrate_state = {std::move(plans),           (unsigned int)nr_baselines,
                       (unsigned int)nr_timesteps, (unsigned int)nr_channels,
                       std::move(wavenumbers),     std::move(visibilities),
                       std::move(weights),         std::move(uvw),
                       std::move(baselines),       std::move(subgrids),
                       std::move(phasors),         std::move(max_nr_timesteps)};
}

void CPU::do_calibrate_update(
    const int antenna_nr, const Array5D<Matrix2x2<std::complex<float>>> &aterms,
    const Array5D<Matrix2x2<std::complex<float>>> &aterm_derivatives,
    Array4D<double> &hessian, Array3D<double> &gradient,
    Array1D<double> &residual) {
  if (m_calibrate_state.plans.empty()) {
    throw std::runtime_error("Calibration was not initialized. Can not update");
  }

  // Arguments
  auto nr_subgrids = m_calibrate_state.plans[antenna_nr][0]->get_nr_subgrids();
  auto nr_channels = m_calibrate_state.wavenumbers.get_x_dim();
  auto nr_terms = aterm_derivatives.get_c_dim();
  auto subgrid_size = aterms.get_a_dim();
  auto nr_stations = aterms.get_c_dim();
  auto nr_timeslots = aterms.get_d_dim();
  auto nr_polarizations = m_grid->get_z_dim();
  auto grid_size = m_grid->get_y_dim();
  auto image_size = grid_size * m_cache_state.cell_size;
  auto w_step = m_cache_state.w_step;

  // Performance measurement
  if (antenna_nr == 0) {
    m_report->initialize(nr_channels, subgrid_size, 0, nr_terms);
  }

  // Data pointers
  auto shift_ptr = m_cache_state.shift.data();
  auto wavenumbers_ptr = m_calibrate_state.wavenumbers.data();
  auto aterm_ptr = reinterpret_cast<std::complex<float> *>(aterms.data());
  auto aterm_derivative_ptr =
      reinterpret_cast<std::complex<float> *>(aterm_derivatives.data());
  auto aterm_idx_ptr =
      m_calibrate_state.plans[antenna_nr][0]->get_aterm_indices_ptr();
  auto metadata_ptr =
      m_calibrate_state.plans[antenna_nr][0]->get_metadata_ptr();
  auto uvw_ptr = m_calibrate_state.uvw.data(antenna_nr);
  auto visibilities_ptr = reinterpret_cast<std::complex<float> *>(
      m_calibrate_state.visibilities.data(antenna_nr));
  float *weights_ptr = (float *)m_calibrate_state.weights.data(antenna_nr);
  auto *subgrids_ptr = m_calibrate_state.subgrids[antenna_nr].data();
  auto *phasors_ptr = m_calibrate_state.phasors[antenna_nr].data();
  double *hessian_ptr = hessian.data();
  double *gradient_ptr = gradient.data();
  double *residual_ptr = residual.data();

  int max_nr_timesteps = m_calibrate_state.max_nr_timesteps[antenna_nr];

  // Run calibration update step
  m_kernels->run_calibrate(
      nr_subgrids, nr_polarizations, grid_size, subgrid_size, image_size,
      w_step, shift_ptr, max_nr_timesteps, nr_channels, nr_terms, nr_stations,
      nr_timeslots, uvw_ptr, wavenumbers_ptr, visibilities_ptr, weights_ptr,
      aterm_ptr, aterm_derivative_ptr, aterm_idx_ptr, metadata_ptr,
      subgrids_ptr, phasors_ptr, hessian_ptr, gradient_ptr, residual_ptr);

  // Performance reporting
  auto current_nr_subgrids = nr_subgrids;
  auto current_nr_timesteps =
      m_calibrate_state.plans[antenna_nr][0]->get_nr_timesteps();
  auto current_nr_visibilities = current_nr_timesteps * nr_channels;
  m_report->update_total(current_nr_subgrids, current_nr_timesteps,
                         current_nr_visibilities);
}

void CPU::do_calibrate_finish() {
  // Performance reporting
  auto nr_antennas = m_calibrate_state.plans.size();
  auto total_nr_timesteps = 0;
  auto total_nr_subgrids = 0;
  for (unsigned int antenna_nr = 0; antenna_nr < nr_antennas; antenna_nr++) {
    total_nr_timesteps +=
        m_calibrate_state.plans[antenna_nr][0]->get_nr_timesteps();
    total_nr_subgrids +=
        m_calibrate_state.plans[antenna_nr][0]->get_nr_subgrids();
  }
  m_report->print_total(nr_correlations, total_nr_timesteps, total_nr_subgrids);
  m_report->print_visibilities(auxiliary::name_calibrate);
}

void CPU::do_transform(DomainAtoDomainB direction) {
#if defined(DEBUG)
  std::cout << __func__ << std::endl;
  std::cout << "FFT (direction: " << direction << ")" << std::endl;
#endif

  try {
    const auto &grid = get_final_grid();

    // Constants
    unsigned int nr_w_layers = grid->get_w_dim();
    unsigned int nr_correlations = grid->get_z_dim();
    unsigned int grid_size = grid->get_y_dim();

    // Performance measurement
    m_report->initialize(0, 0, grid_size);
    m_kernels->set_report(m_report);

    for (unsigned int w = 0; w < nr_w_layers; w++) {
      int sign = (direction == FourierDomainToImageDomain) ? 1 : -1;

      // Grid pointer
      idg::Array3D<std::complex<float>> grid_ptr(grid->data(w), nr_correlations,
                                                 grid_size, grid_size);

      // Constants
      auto grid_size = grid->get_x_dim();

      State states[2];
      states[0] = m_powersensor->read();

      // FFT shift
      if (direction == FourierDomainToImageDomain) {
        m_kernels->shift(grid_ptr);  // TODO: integrate into adder?
      } else {
        m_kernels->shift(grid_ptr);  // TODO: remove
      }

      // Run FFT
      m_kernels->run_fft(grid_size, grid_size, nr_correlations, grid->data(),
                         sign);

      // FFT shift
      if (direction == FourierDomainToImageDomain)
        m_kernels->shift(grid_ptr);  // TODO: remove
      else
        m_kernels->shift(grid_ptr);  // TODO: integrate into splitter?

      // End measurement
      states[1] = m_powersensor->read();
      m_report->update(Report::host, states[0], states[1]);
    }

    // Report performance
    m_report->print_total(nr_correlations);
    std::clog << std::endl;

  } catch (const std::exception &e) {
    std::cerr << __func__ << " caught exception: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << __func__ << " caught unknown exception" << std::endl;
  }
}  // end transform

void CPU::do_compute_avg_beam(
    const unsigned int nr_antennas, const unsigned int nr_channels,
    const Array2D<UVW<float>> &uvw,
    const Array1D<std::pair<unsigned int, unsigned int>> &baselines,
    const Array4D<Matrix2x2<std::complex<float>>> &aterms,
    const Array1D<unsigned int> &aterms_offsets, const Array4D<float> &weights,
    idg::Array4D<std::complex<float>> &average_beam) {
#if defined(DEBUG)
  std::cout << __func__ << std::endl;
#endif

  const unsigned int nr_aterms = aterms_offsets.size() - 1;
  const unsigned int nr_baselines = baselines.get_x_dim();
  const unsigned int nr_timesteps = uvw.get_x_dim();
  const unsigned int subgrid_size = average_beam.get_w_dim();
  const unsigned int nr_polarizations = 4;

  m_report->initialize();
  m_kernels->set_report(m_report);

  auto *baselines_ptr = reinterpret_cast<idg::Baseline *>(baselines.data());
  auto *aterms_ptr = reinterpret_cast<std::complex<float> *>(aterms.data());

  m_kernels->run_average_beam(
      nr_baselines, nr_antennas, nr_timesteps, nr_channels, nr_aterms,
      subgrid_size, nr_polarizations, uvw.data(), baselines_ptr, aterms_ptr,
      aterms_offsets.data(), weights.data(), average_beam.data());

  m_report->print_total(nr_correlations);
}  // end compute_avg_beam

}  // namespace cpu
}  // namespace proxy
}  // namespace idg
