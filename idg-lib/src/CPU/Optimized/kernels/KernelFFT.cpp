// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <iostream>
#include <complex>

#include <math.h>
#include <fftw3.h>
#include <stdint.h>

#include "common/Types.h"
#include "common/Index.h"

#include "idg-config.h"

void copy_1d(unsigned long size, int pol,
             int row,  // copy_row: > 0, copy_col: -1
             int col,  // copy_col: > 0, copy_row: -1
             int dir,  // backward: -1, forward: 1
             std::complex<float>* a, std::complex<float>* b,
             std::complex<float> scale = std::complex<float>(1, 1)) {
  for (unsigned int i = 0; i < size; i++) {
    unsigned long a_idx = row == -1 ? index_grid(size, pol, i, col)
                                    : index_grid(size, pol, row, i);
    unsigned long b_idx = i;
    std::complex<float>& src = dir == 1 ? a[a_idx] : b[b_idx];
    std::complex<float>* dst = dir == 1 ? &b[b_idx] : &a[a_idx];
    *dst = std::complex<float>(src.real() * scale.real(),
                               src.imag() * scale.imag());
  }
}

extern "C" {

void kernel_fft_grid(long size, std::complex<float>* data,
                     int sign  // -1=FFTW_FORWARD, 1=FFTW_BACKWARD
) {
  // Create plan
  fftwf_plan plan;
  fftwf_complex* data_ptr = reinterpret_cast<fftwf_complex*>(data);
  plan = fftwf_plan_dft_1d(size, data_ptr, data_ptr, sign, FFTW_ESTIMATE);

#pragma omp parallel
  {
    // Allocate temporary buffer
    std::vector<std::complex<float>> tmp(size);
    fftwf_complex* tmp_ptr = reinterpret_cast<fftwf_complex*>(tmp.data());

    for (unsigned int pol = 0; pol < NR_POLARIZATIONS; pol++) {
      // Execute 1D FFT over all rows
#pragma omp for
      for (unsigned int y = 0; y < size; y++) {
        // Copy row data -> tmp
        unsigned int x = -1;
        int dir = 1;
        copy_1d(size, pol, y, x, dir, data, tmp.data());

        // Perform the 1D FFT
        fftwf_execute_dft(plan, tmp_ptr, tmp_ptr);

        // Copy row tmp -> data
        dir = -1;
        copy_1d(size, pol, y, x, dir, data, tmp.data());
      }  // end for x

      // Execute 1D FFT over all rows
#pragma omp for
      for (unsigned int x = 0; x < size; x++) {
        // Copy column data -> tmp
        unsigned int y = -1;
        int dir = 1;
        copy_1d(size, pol, y, x, dir, data, tmp.data());

        // Perform the 1D FFT
        fftwf_execute_dft(plan, tmp_ptr, tmp_ptr);

        // Scaling in case of an inverse FFT, so that FFT(iFFT())=identity()
        // => scale by 1/(N*N); since we only use half of the visibilities
        // scale real part by two, and set imaginary part to zero
        std::complex<float> scale(1.0f, 1.0f);
        if (sign == FFTW_BACKWARD) {
          scale = std::complex<float>(2.0f / (size * size), 0);
        }

        // Copy column tmp -> data
        dir = -1;
        copy_1d(size, pol, y, x, dir, data, tmp.data(), scale);
      }
    }  // end for pol
  }    // end omp parallel

  // Destroy plan
  fftwf_destroy_plan(plan);
}

void kernel_fft_subgrid(long size, long batch, fftwf_complex* _data, int sign) {
  fftwf_complex* data = (fftwf_complex*)_data;

  // 2D FFT
  int rank = 2;

  // For grids of size*size elements
  int n[] = {(int)size, (int)size};

  // Set stride
  int istride = 1;
  int ostride = istride;

  // Set dist
  int idist = n[0] * n[1];
  int odist = idist;

  // Planner flags
  int flags = FFTW_ESTIMATE;

  // Create plan
  fftwf_plan plan;
  fftwf_plan_with_nthreads(1);
  plan = fftwf_plan_many_dft(rank, n, NR_POLARIZATIONS, _data, n, istride,
                             idist, _data, n, ostride, odist, sign, flags);

#pragma omp parallel for private(data)
  for (int i = 0; i < batch; i++) {
    data = (fftwf_complex*)_data + i * (NR_POLARIZATIONS * size * size);

    // Execute FFTs
    fftwf_execute_dft(plan, data, data);

    // Scaling in case of an inverse FFT, so that FFT(iFFT())=identity()
    if (sign == FFTW_BACKWARD) {
      float scale = 1 / (double(size) * double(size));
      for (int i = 0; i < NR_POLARIZATIONS * size * size; i++) {
        data[i][0] *= scale;
        data[i][1] *= scale;
      }
    }

  }  // end for batch

  // Cleanup
  fftwf_destroy_plan(plan);
}

void kernel_fft(long gridsize, long size, long batch, fftwf_complex* data,
                int sign) {
  if (size == gridsize) {  // a bit of a hack; TODO: make separate functions for
                           // two cases
    kernel_fft_grid(size, reinterpret_cast<std::complex<float>*>(data), sign);
  } else {
    kernel_fft_subgrid(size, batch, data, sign);
  }
}
}