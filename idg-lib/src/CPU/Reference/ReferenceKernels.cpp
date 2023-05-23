#include "ReferenceKernels.h"
#include "kernels/Kernels.h"

namespace idg {
namespace kernel {
namespace cpu {

using namespace idg::kernel::cpu::reference;

void ReferenceKernels::run_gridder(KERNEL_GRIDDER_ARGUMENTS) {
  pmt::State states[2];
  states[0] = power_meter_->Read();
  kernel_gridder(nr_subgrids, nr_polarizations, grid_size, subgrid_size,
                 image_size, w_step_in_lambda, shift, nr_correlations,
                 nr_channels, nr_stations, uvw, wavenumbers, visibilities,
                 taper, aterms, aterm_indices, avg_aterm, metadata, subgrid);
  states[1] = power_meter_->Read();
  if (report_) {
    report_->update(Report::gridder, states[0], states[1]);
  }
}

void ReferenceKernels::run_degridder(KERNEL_DEGRIDDER_ARGUMENTS) {
  pmt::State states[2];
  states[0] = power_meter_->Read();
  kernel_degridder(nr_subgrids, nr_polarizations, grid_size, subgrid_size,
                   image_size, w_step_in_lambda, shift, nr_correlations,
                   nr_channels, nr_stations, uvw, wavenumbers, visibilities,
                   taper, aterms, aterm_indices, metadata, subgrid);
  states[1] = power_meter_->Read();
  if (report_) {
    report_->update(Report::degridder, states[0], states[1]);
  }
}

void ReferenceKernels::run_average_beam(KERNEL_AVERAGE_BEAM_ARGUMENTS) {
  pmt::State states[2];
  states[0] = power_meter_->Read();
  kernel_average_beam(nr_baselines, nr_antennas, nr_timesteps, nr_channels,
                      nr_aterms, subgrid_size, nr_polarizations, uvw, baselines,
                      aterms, aterm_offsets, weights, average_beam);
  states[1] = power_meter_->Read();
  if (report_) {
    report_->update(Report::average_beam, states[0], states[1]);
  }
}

void ReferenceKernels::run_fft(KERNEL_FFT_ARGUMENTS) {
  pmt::State states[2];
  states[0] = power_meter_->Read();
  kernel_fft(grid_size, size, batch, data, sign);
  states[1] = power_meter_->Read();
  if (report_) {
    report_->update(Report::grid_fft, states[0], states[1]);
  }
}

void ReferenceKernels::run_subgrid_fft(KERNEL_SUBGRID_FFT_ARGUMENTS) {
  pmt::State states[2];
  states[0] = power_meter_->Read();
  kernel_fft(grid_size, size, batch, data, sign);
  states[1] = power_meter_->Read();
  if (report_) {
    report_->update(Report::subgrid_fft, states[0], states[1]);
  }
}

void ReferenceKernels::run_adder(KERNEL_ADDER_ARGUMENTS) {
  pmt::State states[2];
  states[0] = power_meter_->Read();
  kernel_adder(nr_subgrids, nr_polarizations, grid_size, subgrid_size, metadata,
               subgrid, grid);
  states[1] = power_meter_->Read();
  if (report_) {
    report_->update(Report::adder, states[0], states[1]);
  }
}

void ReferenceKernels::run_splitter(KERNEL_SPLITTER_ARGUMENTS) {
  pmt::State states[2];
  states[0] = power_meter_->Read();
  kernel_splitter(nr_subgrids, nr_polarizations, grid_size, subgrid_size,
                  metadata, subgrid, grid);
  states[1] = power_meter_->Read();
  if (report_) {
    report_->update(Report::splitter, states[0], states[1]);
  }
}

}  // namespace cpu
}  // namespace kernel
}  // namespace idg