// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

void compute_extrapolation_scalar(
        int *offset,
  const int outer_dim,
  const int inner_dim,
        float *input_real,
        float *input_imag,
  const float *delta_real,
  const float *delta_imag,
        float *output_real,
        float *output_imag)
{
  for (int o = *offset; o < outer_dim; o++) {
    for (int i = 0; i < inner_dim; i++) {
      float value_current_real = input_real[i];
      float value_current_imag = input_imag[i];
      output_real[i * outer_dim + o] = value_current_real;
      output_imag[i * outer_dim + o] = value_current_imag;
      float value_next_real = 0;
      float value_next_imag = 0;
      value_next_real  = value_current_real * delta_real[i];
      value_next_imag  = value_current_real * delta_imag[i];
      value_next_real -= value_current_imag * delta_imag[i];
      value_next_imag += value_current_imag * delta_real[i];
      input_real[i] = value_next_real;
      input_imag[i] = value_next_imag;
    }
  }

  *offset = outer_dim;
}

void compute_extrapolation(
  const int outer_dim,
  const int inner_dim,
        float *input_real,
        float *input_imag,
  const float *delta_real,
  const float *delta_imag,
        float *output_real,
        float *output_imag)
{
  int offset = 0;

  compute_extrapolation_scalar(
    &offset, outer_dim, inner_dim,
    input_real, input_imag, delta_real, delta_imag,
    output_real, output_imag);
}