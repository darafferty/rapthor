#include "ReferenceKernels.h"
#include "kernels/Kernels.h"

namespace idg {
namespace kernel {
namespace cpu {

using namespace idg::kernel::cpu::reference;

void ReferenceKernels::run_gridder(KERNEL_GRIDDER_ARGUMENTS) {
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

void ReferenceKernels::run_degridder(KERNEL_DEGRIDDER_ARGUMENTS) {
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

void ReferenceKernels::run_average_beam(KERNEL_AVERAGE_BEAM_ARGUMENTS) {
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

void ReferenceKernels::run_fft(KERNEL_FFT_ARGUMENTS) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();
  kernel_fft(grid_size, size, batch, data, sign);
  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::grid_fft, states[0], states[1]);
  }
}

void ReferenceKernels::run_subgrid_fft(KERNEL_SUBGRID_FFT_ARGUMENTS) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();
  kernel_fft(grid_size, size, batch, data, sign);
  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::subgrid_fft, states[0], states[1]);
  }
}

void ReferenceKernels::run_adder(KERNEL_ADDER_ARGUMENTS) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();
  kernel_adder(nr_subgrids, nr_polarizations, grid_size, subgrid_size, metadata,
               subgrid, grid);
  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::adder, states[0], states[1]);
  }
}

void ReferenceKernels::run_splitter(KERNEL_SPLITTER_ARGUMENTS) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();
  kernel_splitter(nr_subgrids, nr_polarizations, grid_size, subgrid_size,
                  metadata, subgrid, grid);
  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::splitter, states[0], states[1]);
  }
}

}  // namespace cpu
}  // namespace kernel
}  // namespace idg