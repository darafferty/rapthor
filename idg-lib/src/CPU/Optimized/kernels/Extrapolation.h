// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

void compute_extrapolation(
  const int nr_channels,
  const int nr_timesteps,
  float *phasor_c_real_,
  float *phasor_c_imag_,
  float *phasor_d_real_,
  float *phasor_d_imag_,
  float *phasor_real,
  float *phasor_imag
)
{
  for (int chan = 0; chan < nr_channels; chan++) {
    for (int time = 0; time < nr_timesteps; time++) {
      float phasor_c_real = phasor_c_real_[time];
      float phasor_c_imag = phasor_c_imag_[time];
      float phasor_d_real = phasor_d_real_[time];
      float phasor_d_imag = phasor_d_imag_[time];
      phasor_real[time * nr_channels + chan] = phasor_c_real;
      phasor_imag[time * nr_channels + chan] = phasor_c_imag;
      float phasor_c_real_next = 0;
      float phasor_c_imag_next = 0;
      phasor_c_real_next  = phasor_c_real * phasor_d_real;
      phasor_c_imag_next  = phasor_c_real * phasor_d_imag;
      phasor_c_real_next -= phasor_c_imag * phasor_d_imag;
      phasor_c_imag_next += phasor_c_imag * phasor_d_real;
      phasor_c_real_[time] = phasor_c_real_next;
      phasor_c_imag_[time] = phasor_c_imag_next;
    }
  }
}