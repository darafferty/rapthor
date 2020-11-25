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
  for (int o = 0; o < outer_dim; o++) {
    for (int i = *offset; i < inner_dim; i++) {
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

  *offset = inner_dim;
}

void compute_extrapolation_avx(
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
  const int vector_length = 8;

  for (int o = 0; o < outer_dim; o++) {
    for (int i = *offset; i < (inner_dim / vector_length) * vector_length; i += vector_length) {
      for (int ii = 0; ii < vector_length; ii++) {
        output_real[(i + ii) * outer_dim + o] = input_real[i + ii];
        output_imag[(i + ii) * outer_dim + o] = input_imag[i + ii];
      }
      __m256 value_current_r = _mm256_load_ps(&input_real[i]);
      __m256 value_current_i = _mm256_load_ps(&input_imag[i]);
      __m256 delta_r = _mm256_load_ps(&delta_real[i]);
      __m256 delta_i = _mm256_load_ps(&delta_imag[i]);
      __m256 value_next_r = _mm256_mul_ps(value_current_r, delta_r);
      __m256 value_next_i = _mm256_mul_ps(value_current_r, delta_i);;
      value_next_r = _mm256_sub_ps(value_next_r, _mm256_mul_ps(value_current_i, delta_i));
      value_next_i = _mm256_add_ps(value_next_i, _mm256_mul_ps(value_current_i, delta_r));
      _mm256_store_ps(&input_real[i], value_next_r);
      _mm256_store_ps(&input_imag[i], value_next_i);
    }
  }

  *offset += vector_length * ((inner_dim - *offset) / vector_length);
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

  compute_extrapolation_avx(
    &offset, outer_dim, inner_dim,
    input_real, input_imag, delta_real, delta_imag,
    output_real, output_imag);

  compute_extrapolation_scalar(
    &offset, outer_dim, inner_dim,
    input_real, input_imag, delta_real, delta_imag,
    output_real, output_imag);
}