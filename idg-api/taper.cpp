// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "taper.h"

#include <cmath>
#include <vector>

// #if defined(HAVE_MKL)
//     #include <mkl_lapacke.h>
// #else
//     // Workaround: Prevent c-linkage of templated complex<double> in
//     lapacke.h #include <complex.h> #define lapack_complex_float    float
//     _Complex #define lapack_complex_double   double _Complex
//     // End workaround
//     #include <lapacke.h>
// #endif

/* DGESVD prototype */
extern "C" void dgesvd_(char* jobu, char* jobvt, int* m, int* n, double* a,
                        int* lda, double* s, double* u, int* ldu, double* vt,
                        int* ldvt, double* work, int* lwork, int* info);

void init_optimal_taper_1D(int subgridsize, int padded_size, int size,
                           float kernelsize, float* taper_subgrid,
                           float* taper_grid) {
  float padding = float(padded_size) / float(size);
  int N = subgridsize;
  float W = kernelsize;

  double* RR = new double[(N * 16 + 1) * N * N];
  double* B = new double[(N * 16 + 1) * N];

  double* Q = new double[N * N];

  for (int i = 0; i < N; i++) {
    Q[i * N + i] = 1.0;
  }

  for (int i = 1; i < N; i++) {
    double x = (M_PI * (N - W + 1) * i) / N;
    double s = std::sin(x) / x;
    for (int j = 0; j < (N - i); j++) {
      Q[(i + j) * N + j] = Q[j * N + (i + j)] = s;
    }
  }

  for (int i = 0; i < N * 16 + 1; i++) {
    float l = (float(i) / N / 16 - 0.5) / padding;

    double d[N];
    double s[N];
    for (int j = 0; j < N; j++) {
      d[j] = 0.0;
      double ll = (j - N / 2 + 0.5) / N - l;
      for (int k = 0; k < N; k++) {
        double u = k - N / 2 + 0.5;
        double phase = 2 * M_PI * u * ll;
        d[j] += std::cos(phase) / N;
      }

      double x = M_PI * (N - W + 1) * ll;
      if (x)
        s[j] = std::sin(x) / x;
      else
        s[j] = 1.0;
      B[i * N + j] = d[j] * s[j];
    }

    double D[N][N];
    double S[N][N];
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        D[j][k] = d[j] * d[k];
        S[j][k] = s[j] * s[k];
        RR[i * N * N + j * N + k] = D[j][k] * (Q[j * N + k] - S[j][k]);
      }
    }
  }

  double R[N / 2][N / 2];
  double S1[N / 2];
  double U1[N / 2][N / 2];
  double V1T[N / 2][N / 2];
  double superb[N / 2 - 1];

  double* taper = new double[N * 16 + 1];

  for (int i = 0; i < N * 16 + 1; i++) {
    taper[i] = 1.0;
  }

  for (int ii = 0; ii < 10; ii++) {
    for (int j = 0; j < N / 2; j++) {
      for (int k = 0; k < N / 2; k++) {
        R[j][k] = 0.0;
      }
    }

    for (int i = 0; i < N * 16 + 1; i++) {
      for (int j = 0; j < N / 2; j++) {
        for (int k = 0; k < N / 2; k++) {
          R[j][k] +=
              (RR[i * N * N + j * N + k] + RR[i * N * N + j * N + N - 1 - k] +
               RR[i * N * N + (N - 1 - j) * N + k] +
               RR[i * N * N + (N - 1 - j) * N + (N - 1 - k)]) /
              (taper[i] * taper[i]);
        }
      }
    }

    //         LAPACKE_dgesvd( LAPACK_COL_MAJOR, 'A', 'A', N/2, N/2, (double*)
    //         R, N/2, (double*) S1, (double*) U1, N/2, (double*) V1T, N/2,
    //         (double*) superb );

    int lwork = -1;
    int m = N / 2;
    int n = N / 2;

    int lda = N / 2;
    int ldu = N / 2;
    int ldvt = N / 2;
    int info;
    double wkopt;
    /* Local arrays */
    double* s = (double*)S1;
    double* u = (double*)U1;
    double* vt = (double*)V1T;
    double* a = (double*)R;

    char job = 'A';

    /* Query and allocate the optimal workspace */
    dgesvd_(&job, &job, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &wkopt, &lwork,
            &info);
    lwork = (int)wkopt;
    std::vector<double> work(lwork);
    /* Compute SVD */
    dgesvd_(&job, &job, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work.data(),
            &lwork, &info);

    //         std::cout << "U:" << std::endl;
    for (int i = 0; i < N / 2; i++) {
      //             std::cout << U1[N/2-1][i];
      //             if (i < N/2-1) std::cout << ", ";
    }
    //         std::cout << std::endl;

    for (int i = 0; i < N * 16 + 1; i++) {
      taper[i] = 0.0;
      for (int j = 0; j < N / 2; j++) {
        taper[i] += (B[i * N + j] + B[i * N + N - 1 - j]) * U1[N / 2 - 1][j];
      }
    }
  }

  for (int i = 0; i < N / 2; i++) {
    taper_subgrid[i] = U1[N / 2 - 1][i] / U1[N / 2 - 1][N / 2 - 1];
    taper_subgrid[N - 1 - i] = U1[N / 2 - 1][i] / U1[N / 2 - 1][N / 2 - 1];
  }

#pragma omp parallel for
  for (int i = 0; i < padded_size; i++) {
    taper_grid[i] = 0.0;
    float l = (float(i) / padded_size - 0.5);
    double d;
    double s;
    for (int j = 0; j < N; j++) {
      d = 0.0;
      double ll = (j - N / 2 + 0.5) / N - l;
      for (int k = 0; k < N; k++) {
        double u = k - N / 2 + 0.5;
        double phase = 2 * M_PI * u * ll;
        d += std::cos(phase) / N;
      }

      double x = M_PI * (N - W + 1) * ll;
      if (x)
        s = std::sin(x) / x;
      else
        s = 1.0;
      taper_grid[i] += taper_subgrid[j] * d * s;
    }
  }

  delete[] RR;
  delete[] B;
  delete[] Q;
  delete[] taper;
}

void init_optimal_gridding_taper_1D(int subgridsize, int gridsize,
                                    float kernelsize, float* taper_subgrid,
                                    float* taper_grid) {
  int N = subgridsize;
  float W = kernelsize;

  double* RR = new double[N * N];

  for (int i = 0; i < N; i++) {
    RR[i * N + i] = 1.0;
  }

  for (int i = 1; i < N; i++) {
    double x = M_PI * W * i / N;
    double s = std::sin(x) / x;
    for (int j = 0; j < (N - i); j++) {
      RR[(i + j) * N + j] = RR[j * N + (i + j)] = s;
    }
  }

  double R[N / 2][N / 2];
  for (int j = 0; j < N / 2; j++) {
    for (int k = 0; k < N / 2; k++) {
      R[j][k] = RR[j * N + k] + RR[j * N + N - 1 - k] +
                RR[(N - 1 - j) * N + k] + RR[(N - 1 - j) * N + (N - 1 - k)];
    }
  }

  delete[] RR;

  double S[N / 2];
  double U[N / 2][N / 2];
  double VT[N / 2][N / 2];
  double superb[N / 2 - 1];

  //     LAPACKE_dgesvd( LAPACK_COL_MAJOR, 'A', 'A', N/2, N/2, (double*) R, N/2,
  //     (double*) S, (double*) U, N/2, (double*) VT, N/2, (double*) superb );

  int lwork = -1;
  int m = N / 2;
  int n = N / 2;

  int lda = N / 2;
  int ldu = N / 2;
  int ldvt = N / 2;
  int info;
  double wkopt;
  /* Local arrays */
  double* s = (double*)S;
  double* u = (double*)U;
  double* vt = (double*)VT;
  double* a = (double*)R;

  char job = 'A';

  /* Query and allocate the optimal workspace */
  dgesvd_(&job, &job, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &wkopt, &lwork,
          &info);
  lwork = (int)wkopt;
  std::vector<double> work(lwork);
  /* Compute SVD */
  dgesvd_(&job, &job, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work.data(),
          &lwork, &info);

  //     std::cout << "U:" << std::endl;
  //     std::cout << std::endl;

  for (int i = 0; i < N / 2; i++) {
    taper_subgrid[i] = taper_subgrid[N - i - 1] = U[0][i] / U[0][N / 2 - 1];
    //         std::cout << taper_subgrid[i];
    //         if (i < N/2) std::cout << ", ";
  }

  for (int i = 0; i < gridsize; i++) {
    taper_grid[i] = 0.0;
    float l = (float(i) / gridsize - 0.5);
    double d;
    double s;
    for (int j = 0; j < N; j++) {
      d = 0.0;
      double ll = (j - N / 2 + 0.5) / N - l;
      for (int k = 0; k < N; k++) {
        double u = k - N / 2 + 0.5;
        double phase = 2 * M_PI * u * ll;
        d += std::cos(phase) / N;
      }

      double x = M_PI * (N - W + 1) * ll;
      if (x)
        s = std::sin(x) / x;
      else
        s = 1.0;
      taper_grid[i] += taper_subgrid[j] * d * s;
    }
  }
}

double bessel0(double x, double precision) {
  // Calculate I_0 = SUM of m 0 -> inf [ (x/2)^(2m) ]
  // This is the unnormalized bessel function of order 0.
  double d = 0.0, ds = 1.0, sum = 1.0;
  do {
    d += 2.0;
    ds *= x * x / (d * d);
    sum += ds;
  } while (ds > sum * precision);
  return sum;
}

void init_kaiser_bessel_1D(int size, float* taper_grid) {
  const int mid = (size + 1) / 2;
  const double alpha = 8.6;
  const double normFactor = 1.0 / bessel0(alpha, 1e-8);

  for (int i = 0; i != mid; i++) {
    double term = 1.0 - (double(i) / mid);
    taper_grid[i] =
        bessel0(alpha * sqrt(1.0 - (term * term)), 1e-10) * normFactor;
    taper_grid[size - i - 1] = taper_grid[i];
  }
}

void init_blackman_harris_1D(int size, float* taper_grid) {
  for (int i = 0; i != size; ++i) {
    const static double a0 = 0.35875, a1 = 0.48829, a2 = 0.14128, a3 = 0.01168;
    const double id = double(i) * 2.0 * M_PI, n = int(size) - 1;
    taper_grid[i] = a0 - a1 * std::cos((1.0 * id) / n) +
                    a2 * std::cos((2.0 * id) / n) -
                    a3 * std::cos((3.0 * id) / n);
  }
}
