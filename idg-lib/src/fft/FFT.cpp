// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "FFT.h"

#include <fftw3.h>

using namespace std;

namespace idg {

void kernel_fft_composite(unsigned batch, int m, int n,
                          std::complex<float> *data, int sign) {
  fftwf_complex *in_ptr = reinterpret_cast<fftwf_complex *>(data);
  fftwf_complex *out_ptr = reinterpret_cast<fftwf_complex *>(data);

  // Initialize FFT plans
  fftwf_plan plan_col = fftwf_plan_dft_1d(n, NULL, NULL, sign, FFTW_ESTIMATE);
  fftwf_plan plan_row =
      m == n ? plan_col : fftwf_plan_dft_1d(m, NULL, NULL, sign, FFTW_ESTIMATE);

  for (unsigned i = 0; i < batch; i++) {
// FFT over rows
#pragma omp parallel for
    for (int y = 0; y < m; y++) {
      uint64_t offset = size_t(i) * size_t(m) * size_t(n) + y * size_t(n);
      fftwf_execute_dft(plan_col, in_ptr + offset, out_ptr + offset);
    }

// Iterate all columns
#pragma omp parallel for
    for (int x = 0; x < n; x++) {
      std::complex<float> tmp[m];

      // Copy column into temporary buffer
      for (int y = 0; y < m; y++) {
        uint64_t offset = size_t(i) * size_t(m) * size_t(n) + y * size_t(n) + x;
        tmp[y] = data[offset];
      }

      // FFT column
      fftwf_complex *tmp_ptr = reinterpret_cast<fftwf_complex *>(tmp);
      fftwf_execute_dft(plan_col, tmp_ptr, tmp_ptr);

      // Store the result in the output buffer
      for (int y = 0; y < m; y++) {
        uint64_t offset = size_t(i) * size_t(m) * size_t(n) + y * size_t(n) + x;
        data[offset] = tmp[y];
      }
    }
  }

  // Free FFT plans
  fftwf_destroy_plan(plan_col);
  if (m != n) {
    fftwf_destroy_plan(plan_row);
  }
}

void kernel_fft_coarse(int batch, int height, int width,
                       std::complex<float> *data, int sign) {
  fftwf_complex *data_ptr = reinterpret_cast<fftwf_complex *>(data);

  // Create plan
  fftwf_plan plan;
  plan =
      fftwf_plan_dft_2d(height, width, data_ptr, data_ptr, sign, FFTW_ESTIMATE);

#pragma omp parallel for private(data_ptr)
  for (int i = 0; i < batch; i++) {
    data_ptr = reinterpret_cast<fftwf_complex *>(data) + (i * height * width);

    // Execute FFTs
    fftwf_execute_dft(plan, data_ptr, data_ptr);
  }  // end for batch

  // Cleanup
  fftwf_destroy_plan(plan);
}

void kernel_fft(unsigned batch, int height, int width,
                std::complex<float> *data, int sign) {
  int n = std::max(height, width);

  // Select FFT based on the size of the transformation
  // On AMD Epyc (Zen 2), performing many small FFTs is faster using 2D
  // FFTs while the composite approach is faster for larger transformations.
  if (n < 256) {
    kernel_fft_coarse(batch, height, width, data, sign);
  } else {
    kernel_fft_composite(batch, height, width, data, sign);
  }
}

void fft2f(unsigned batch, int m, int n, complex<float> *data) {
  ifftshift(batch, m, n, data);
  kernel_fft(batch, m, n, data, FFTW_FORWARD);
  fftshift(batch, m, n, data);
}

void fft2f(int m, int n, std::complex<float> *data) { fft2f(1, m, n, data); }

void fft2f(int n, std::complex<float> *data) { fft2f(n, n, data); }

void ifft2f(unsigned batch, int m, int n, complex<float> *data) {
  ifftshift(batch, m, n, data);
  kernel_fft(batch, m, n, data, FFTW_BACKWARD);
  fftshift(batch, m, n, data);
}

void ifft2f(int m, int n, std::complex<float> *data) { ifft2f(1, n, n, data); }

void ifft2f(int n, std::complex<float> *data) { ifft2f(n, n, data); }

void fft2f_r2c(int m, int n, float *data_in, complex<float> *data_out) {
  fftwf_complex *tmp = (fftwf_complex *)data_out;
  fftwf_plan plan;

#pragma omp critical
  plan = fftwf_plan_dft_r2c_2d(m, n, data_in, tmp, FFTW_ESTIMATE);
  ifftshift(m, n, tmp);
  fftwf_execute(plan);
  fftshift(m, n, tmp);
  fftwf_destroy_plan(plan);
}

void fft2f_r2c(int n, float *data_in, complex<float> *data_out) {
  fft2f_r2c(n, n, data_in, data_out);
}

void ifft2f_c2r(int m, int n, complex<float> *data_in, float *data_out) {
  fftwf_complex *tmp = (fftwf_complex *)data_in;
  fftwf_plan plan;

#pragma omp critical
  plan = fftwf_plan_dft_c2r_2d(m, n, tmp, data_out, FFTW_ESTIMATE);
  ifftshift(m, n, tmp);
  fftwf_execute(plan);
  fftshift(m, n, tmp);
  fftwf_destroy_plan(plan);
}

void ifft2f_c2r(int n, complex<float> *data_in, float *data_out) {
  ifft2f_c2r(n, n, data_in, data_out);
}

void resize2f(int m_in, int n_in, complex<float> *data_in, int m_out, int n_out,
              complex<float> *data_out) {
  // scale before FFT
  float s = 1.0f / (m_in * n_in);
  for (int i = 0; i < m_in; i++) {
    for (int j = 0; j < n_in; j++) {
      data_in[i * n_in + j] *= s;
    }
  }

  // in-place FFT
  fft2f(m_in, n_in, data_in);

  // put FFTed data in center
  int m_offset = int((m_out - m_in) / 2);
  int n_offset = int((n_out - n_in) / 2);
  if (m_offset >= 0 && n_offset >= 0) {
    for (int i = 0; i < m_in; i++) {
      for (int j = 0; j < n_in; j++) {
        data_out[(i + m_offset) * n_out + (j + n_offset)] =
            data_in[i * n_in + j];
      }
    }
  } else if (m_offset < 0 && n_offset < 0) {
    m_offset = int((m_in - m_out) / 2);
    n_offset = int((n_in - n_out) / 2);
    for (int i = 0; i < m_out; i++) {
      for (int j = 0; j < n_out; j++) {
        data_out[i * n_out + j] =
            data_in[(i + m_offset) * n_in + (j + n_offset)];
      }
    }
  } else if (m_offset >= 0 && n_offset < 0) {
    n_offset = int((n_in - n_out) / 2);
    for (int i = 0; i < m_in; i++) {
      for (int j = 0; j < n_out; j++) {
        data_out[(i + m_offset) * n_out + j] =
            data_in[i * n_in + (j + n_offset)];
      }
    }
  } else if (m_offset < 0 && n_offset >= 0) {
    m_offset = int((m_in - m_out) / 2);
    for (int i = 0; i < m_out; i++) {
      for (int j = 0; j < n_in; j++) {
        data_out[i * n_out + (j + n_offset)] =
            data_in[(i + m_offset) * n_in + j];
      }
    }
  }

  // in-place inverse FFT
  ifft2f(m_out, n_out, data_out);
}

void resize2f(int m_in, int n_in, float *data_in, int m_out, int n_out,
              float *data_out) {
  auto copy_in = new std::complex<float>[m_in * n_in];
  auto copy_out = new std::complex<float>[m_out * n_out];

  for (int i = 0; i < m_in; i++) {
    for (int j = 0; j < n_in; j++) {
      copy_in[i * n_in + j] = data_in[i * n_in + j];
    }
  }

  resize2f(m_in, n_in, copy_in, m_out, n_out, copy_out);

  for (int i = 0; i < m_out; i++) {
    for (int j = 0; j < n_out; j++) {
      data_out[i * n_out + j] = copy_out[i * n_out + j].real();
    }
  }

  delete[] copy_in;
  delete[] copy_out;
}

}  // end namespace idg
