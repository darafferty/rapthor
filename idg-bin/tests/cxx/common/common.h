// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/Proxy.h"

#include <complex>

// Enable/disable tests by setting the corresponding definition
#define TEST_GRIDDING 1
#define TEST_DEGRIDDING 1
#define TEST_AVERAGE_BEAM 1

float get_accuracy(const int n, const std::complex<float>* A,
                   const std::complex<float>* B);

void print_parameters(unsigned int nr_stations, unsigned int nr_channels,
                      unsigned int nr_timesteps, unsigned int nr_timeslots,
                      float image_size, unsigned int grid_size,
                      unsigned int subgrid_size, unsigned int kernel_size);

int compare(idg::proxy::Proxy& proxy1, idg::proxy::Proxy& proxy2,
            float tol = 1000 * std::numeric_limits<float>::epsilon());