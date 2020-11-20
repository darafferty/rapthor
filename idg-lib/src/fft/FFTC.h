// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "FFT.h"

using namespace std;

extern "C" {

void fft2f(int m, int n, void* data) {
  idg::fft2f(m, n, (complex<float>*)data);
}

void ifft2f(int m, int n, void* data) {
  idg::ifft2f(m, n, (complex<float>*)data);
}

void fft2f_r2c(int m, int n, void* data_in, void* data_out) {
  idg::fft2f_r2c(m, n, (float*)data_in, (complex<float>*)data_out);
}

void ifft2f_c2r(int m, int n, void* data_in, void* data_out) {
  idg::ifft2f_c2r(m, n, (complex<float>*)data_in, (float*)data_out);
}

void fftshift2f(int m, int n, void* array) {
  idg::fftshift(m, n, (complex<float>*)array);
}

void ifftshift2f(int m, int n, void* array) {
  idg::ifftshift(m, n, (complex<float>*)array);
}

void resize2f_r2r(int m_in, int n_in, void* data_in, int m_out, int n_out,
                  void* data_out) {
  idg::resize2f(m_in, n_in, (float*)data_in, m_out, n_out, (float*)data_out);
}

void resize2f_c2c(int m_in, int n_in, void* data_in, int m_out, int n_out,
                  void* data_out) {
  idg::resize2f(m_in, n_in, (complex<float>*)data_in, m_out, n_out,
                (complex<float>*)data_out);
}
}
