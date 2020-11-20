// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

/*
 *  Wrapper routines to FFTW library
 */

#ifndef IDG_FFTW_H_
#define IDG_FFTW_H_

#include "idg-config.h"
#include <iostream>
#include <complex>
#include <stdexcept>
#include <cstring>

namespace idg {

// in-place 2d-FFT for complex float m-by-n array
void fft2f(int m, int n, std::complex<float> *data);

// in-place 2d-FFT for complex float n-by-n array
void fft2f(int n, std::complex<float> *data);

// in-place batched 2d-FFT for complex float  m-b-n arrays
void fft2f(unsigned batch, int m, int n, std::complex<float> *data);

// in-place 2d-iFFT for complex float m-by-n arrays
void ifft2f(int m, int n, std::complex<float> *data);

// in-place batched 2d-iFFT for complex float  m-b-n arrays
void ifft2f(unsigned batch, int m, int n, std::complex<float> *data);

// in-place 2d-iFFT for complex float n-by-n array
void ifft2f(int n, std::complex<float> *data);

// real-to-complex 2d-FFT for m-by-n array
void fft2f_r2c(int m, int n, float *data_in, std::complex<float> *data_out);

// real-to-complex 2d-FFT for n-by-n array
void fft2f_r2c(int n, float *data_in, std::complex<float> *data_out);

// complex-to-real 2d-FFT for m-by-n array
void ifft2f_c2r(int m, int n, std::complex<float> *data_in, float *data_out);

// complex-to-real 2d-FFT for n-by-n array
void ifft2f_c2r(int n, std::complex<float> *data_in, float *data_out);

// fftshift for m-by-n array of type T
// TODO: make work for odd dimensions
template <typename T>
void fftshift(int m, int n, T *array) {
  auto buffer = new T[n];

  for (int i = 0; i < m / 2; i++) {
    // save i-th row into buffer
    memcpy(buffer, &array[i * n], n * sizeof(T));

    auto j = i + m / 2;
    std::memcpy(&array[i * n + n / 2], &array[j * n], (n / 2) * sizeof(T));
    std::memcpy(&array[i * n], &array[j * n + n / 2], (n / 2) * sizeof(T));
    std::memcpy(&array[j * n], &buffer[n / 2], (n / 2) * sizeof(T));
    std::memcpy(&array[j * n + n / 2], &buffer[0], (n / 2) * sizeof(T));
  }

  delete[] buffer;
}

// fftshift for m-by-n array of type T
// TODO: make work for odd dimensions
template <typename T>
void fftshift(int batch, int m, int n, T *array) {
#pragma omp parallel for
  for (size_t i = 0; i < batch; i++) {
    fftshift(m, n, &array[i * m * n]);
  }
}

// fftshift for n-by-n array of type T
// TODO: make work for odd dimensions
template <typename T>
void fftshift(int n, T *array) {
  fftshift(n, n, array);
}

// ifftshift for m-by-n array of type T
// TODO: make work for odd dimensions
template <typename T>
void ifftshift(int m, int n, T *array) {
  if (n % 2 != 0) throw std::invalid_argument("Only square grids supported.");
  fftshift(m, n, array);
}

// ifftshift for n-by-n array of type T
// TODO: make work for odd dimensions
template <typename T>
void ifftshift(int n, T *array) {
  ifftshift(n, n, array);
}

// resize 2-dimensional array to (larger) size
void resize2f(int m_in, int n_in, std::complex<float> *data_in, int m_out,
              int n_out, std::complex<float> *data_out);
void resize2f(int m_in, int n_in, float *data_in, int m_out, int n_out,
              float *data_out);
void resize2(int m_in, int n_in, std::complex<double> *data_in, int m_out,
             int n_out, std::complex<double> *data_out);
void resize2(int m_in, int n_in, double *data_in, int m_out, int n_out,
             double *data_out);

}  // namespace idg

#endif
