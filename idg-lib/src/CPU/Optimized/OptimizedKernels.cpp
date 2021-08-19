#include "../Reference/ReferenceKernels.h"
#include "OptimizedKernels.h"
#include "kernels/Kernels.h"

namespace idg {
namespace kernel {
namespace cpu {

using namespace idg::kernel::cpu::reference;
using namespace idg::kernel::cpu::optimized;

/*
 * Main
 */
void OptimizedKernels::run_gridder(KERNEL_GRIDDER_ARGUMENTS) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();
  kernel_gridder(nr_subgrids, nr_polarizations, grid_size, subgrid_size,
                 image_size, w_step_in_lambda, shift, nr_correlations,
                 nr_channels, nr_stations, uvw, wavenumbers, visibilities,
                 spheroidal, aterms, aterms_indices, avg_aterm, metadata,
                 subgrid);
  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::gridder, states[0], states[1]);
  }
}

void OptimizedKernels::run_degridder(KERNEL_DEGRIDDER_ARGUMENTS) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();
  kernel_degridder(nr_subgrids, nr_polarizations, grid_size, subgrid_size,
                   image_size, w_step_in_lambda, shift, nr_correlations,
                   nr_channels, nr_stations, uvw, wavenumbers, visibilities,
                   spheroidal, aterms, aterms_indices, metadata, subgrid);
  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::degridder, states[0], states[1]);
  }
}

void OptimizedKernels::run_average_beam(KERNEL_AVERAGE_BEAM_ARGUMENTS) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();
  kernel_average_beam(nr_baselines, nr_antennas, nr_timesteps, nr_channels,
                      nr_aterms, subgrid_size, nr_polarizations, uvw, baselines,
                      aterms, aterms_offsets, weights, average_beam);
  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::average_beam, states[0], states[1]);
  }
}

void OptimizedKernels::run_fft(KERNEL_FFT_ARGUMENTS) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();
  kernel_fft(grid_size, size, batch, data, sign);
  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::grid_fft, states[0], states[1]);
  }
}

void OptimizedKernels::run_subgrid_fft(KERNEL_SUBGRID_FFT_ARGUMENTS) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();
  kernel_fft(grid_size, size, batch, data, sign);
  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::subgrid_fft, states[0], states[1]);
  }
}

void OptimizedKernels::run_adder(KERNEL_ADDER_ARGUMENTS) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();
  kernel_adder(nr_subgrids, nr_polarizations, grid_size, subgrid_size, metadata,
               subgrid, grid);
  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::adder, states[0], states[1]);
  }
}

void OptimizedKernels::run_splitter(KERNEL_SPLITTER_ARGUMENTS) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();
  kernel_splitter(nr_subgrids, nr_polarizations, grid_size, subgrid_size,
                  metadata, subgrid, grid);
  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::splitter, states[0], states[1]);
  }
}

/*
 * Calibration
 */
void OptimizedKernels::run_calibrate(KERNEL_CALIBRATE_ARGUMENTS) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();
  kernel_calibrate(nr_subgrids, nr_polarizations, grid_size, subgrid_size,
                   image_size, w_step_in_lambda, shift, max_nr_timesteps,
                   nr_channels, nr_stations, nr_terms, nr_time_slots, uvw,
                   wavenumbers, visibilities, weights, aterms,
                   aterm_derivatives, aterms_indices, metadata, subgrid,
                   phasors, hessian, gradient, residual);
  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update<Report::calibrate>(states[0], states[1]);
  }
}

void OptimizedKernels::run_calibrate_phasor(KERNEL_CALIBRATE_PHASOR_ARGUMENTS) {
  kernel_phasor(nr_subgrids, grid_size, subgrid_size, image_size,
                w_step_in_lambda, shift, max_nr_timesteps, nr_channels, uvw,
                wavenumbers, metadata, phasors);
}

/*
 * W-Stacking
 */
void OptimizedKernels::run_adder_wstack(KERNEL_ADDER_WSTACK_ARGUMENTS) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();
  kernel_adder_wstack(nr_subgrids, nr_polarizations, grid_size, subgrid_size,
                      metadata, subgrid, grid);
  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::adder, states[0], states[1]);
  }
}

void OptimizedKernels::run_splitter_wstack(KERNEL_SPLITTER_WSTACK_ARGUMENTS) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();
  kernel_splitter_wstack(nr_subgrids, nr_polarizations, grid_size, subgrid_size,
                         metadata, subgrid, grid);
  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::splitter, states[0], states[1]);
  }
}

/*
 * W-Tiling
 */
size_t OptimizedKernels::init_wtiles(int nr_polarizations, size_t grid_size,
                                     int subgrid_size) {
  // Heuristic for choosing the number of wtiles.
  // A number that is too small will result in excessive flushing, too large in
  // excessive memory usage.
  //
  // Current heuristic is for the wtiles to cover 50% the grid.
  // Because of padding with subgrid_size, the memory used will be more than 50%
  // of the memory used for the grid. In the extreme case subgrid_size is
  // equal to kWTileSize (both 128) m_wtiles_buffer will be as large as the
  // grid. The minimum number of wtiles is 4.
  size_t nr_wtiles_min = 4;
  size_t nr_wtiles = std::max(
      nr_wtiles_min, (grid_size * grid_size) / (kWTileSize * kWTileSize) / 2);

  // Make sure that the wtiles buffer does not use an excessive amount of memory
  const size_t padded_wtile_size = size_t(kWTileSize) + size_t(subgrid_size);
  size_t sizeof_padded_wtile = nr_polarizations * padded_wtile_size *
                               padded_wtile_size * sizeof(std::complex<float>);
  size_t sizeof_padded_wtiles = nr_wtiles * sizeof_padded_wtile;
  size_t free_memory = auxiliary::get_free_memory() * 1024 * 1024;  // Bytes
  while (sizeof_padded_wtiles > free_memory) {
    nr_wtiles *= 0.9;
    sizeof_padded_wtiles = nr_wtiles * sizeof_padded_wtiles;
  }
  assert(nr_wtiles >= nr_wtiles_min);

  m_wtiles_buffer = idg::Array1D<std::complex<float>>(
      sizeof_padded_wtiles / sizeof(std::complex<float>));
  m_wtiles_buffer.zero();
  return nr_wtiles;
}

void OptimizedKernels::run_adder_tiles_to_grid(
    KERNEL_ADDER_TILES_TO_GRID_ARGUMENTS) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();
  kernel_adder_wtiles_to_grid(nr_polarizations, grid_size, subgrid_size,
                              kWTileSize, image_size, w_step, shift, nr_tiles,
                              tile_ids, tile_coordinates,
                              m_wtiles_buffer.data(), grid);
  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::wtiling_forward, states[0], states[1]);
  }
}

void OptimizedKernels::run_adder_wtiles(KERNEL_ADDER_WTILES_ARGUMENTS) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();
  for (int subgrid_index = 0; subgrid_index < (int)nr_subgrids;) {
    // Is a flush needed right now?
    if (!wtile_flush_set.empty() && wtile_flush_set.front().subgrid_index ==
                                        subgrid_index + subgrid_offset) {
      // Get information on what wtiles to flush
      WTileUpdateInfo &wtile_flush_info = wtile_flush_set.front();

      // Project wtiles to master grid
      kernel_adder_wtiles_to_grid(nr_polarizations, grid_size, subgrid_size,
                                  kWTileSize, image_size, w_step, shift,
                                  wtile_flush_info.wtile_ids.size(),
                                  wtile_flush_info.wtile_ids.data(),
                                  wtile_flush_info.wtile_coordinates.data(),
                                  m_wtiles_buffer.data(), grid);

      // Remove the flush event from the queue
      wtile_flush_set.pop_front();
    }

    // Initialize number of subgrids to process next to all remaining subgrids
    // in job
    int nr_subgrids_to_process = nr_subgrids - subgrid_index;

    // Check whether a flush needs to happen before the end of the job
    if (!wtile_flush_set.empty() && wtile_flush_set.front().subgrid_index -
                                            (subgrid_index + subgrid_offset) <
                                        nr_subgrids_to_process) {
      // Reduce the number of subgrids to process to just before the next flush
      // event
      nr_subgrids_to_process = wtile_flush_set.front().subgrid_index -
                               (subgrid_index + subgrid_offset);
    }

    // Add all subgrids than can be added to the wtiles
    kernel_adder_subgrids_to_wtiles(nr_subgrids_to_process, nr_polarizations,
                                    grid_size, subgrid_size, kWTileSize,
                                    &metadata[subgrid_index],
                                    &subgrid[subgrid_index * subgrid_size *
                                             subgrid_size * nr_polarizations],
                                    m_wtiles_buffer.data());
    // Increment the subgrid index by the actual number of processed subgrids
    subgrid_index += nr_subgrids_to_process;
  }

  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::wtiling_forward, states[0], states[1]);
  }
}

void OptimizedKernels::run_splitter_wtiles(KERNEL_SPLITTER_WTILES_ARGUMENTS) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();

  for (int subgrid_index = 0; subgrid_index < nr_subgrids;) {
    // Check whether initialize is needed right now
    if (!wtile_initialize_set.empty() &&
        wtile_initialize_set.front().subgrid_index ==
            (int)(subgrid_index + subgrid_offset)) {
      // Get the information on what wtiles to initialize
      WTileUpdateInfo &wtile_initialize_info = wtile_initialize_set.front();
      // Initialize the wtiles from the grid
      kernel_splitter_wtiles_from_grid(
          nr_polarizations, grid_size, subgrid_size, kWTileSize, image_size,
          w_step, shift, wtile_initialize_info.wtile_ids.size(),
          wtile_initialize_info.wtile_ids.data(),
          wtile_initialize_info.wtile_coordinates.data(),
          m_wtiles_buffer.data(), grid);

      // Remove initialize even from queue
      wtile_initialize_set.pop_front();
    }

    // Initialize number of subgrids to proccess next to all remaining subgrids
    // in job
    int nr_subgrids_to_process = nr_subgrids - subgrid_index;

    // Check whether initialization needs to happen before the end of the job
    if (!wtile_initialize_set.empty() &&
        wtile_initialize_set.front().subgrid_index -
                (subgrid_index + subgrid_offset) <
            nr_subgrids_to_process) {
      // Reduce the number of subgrids to process to just before the next
      // initialization event
      nr_subgrids_to_process = wtile_initialize_set.front().subgrid_index -
                               (subgrid_offset + subgrid_index);
    }

    // Process all subgrids that can be processed now
    kernel_splitter_subgrids_from_wtiles(
        nr_subgrids_to_process, nr_polarizations, grid_size, subgrid_size,
        kWTileSize, &metadata[subgrid_index],
        &subgrid[subgrid_index * subgrid_size * subgrid_size *
                 nr_polarizations],
        m_wtiles_buffer.data());

    // Increment the subgrid index by the actual number of processed subgrids
    subgrid_index += nr_subgrids_to_process;
  }  // end for subgrid_index

  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::wtiling_backward, states[0], states[1]);
  }
}  // end run_splitter_wtiles

}  // namespace cpu
}  // namespace kernel
}  // namespace idg