
#include <iostream>
#include <complex>

#include <math.h>
#include <fftw3.h>
#include <stdint.h>

#include "Types.h"

#include "idg-config.h"


extern "C" {
void kernel_fft_grid(
	long size,
    fftwf_complex *_data,
    int sign    // -1=FFTW_FORWARD, 1=FFTW_BACKWARD
    ) {
    // Use multiple threads for each polarization
    #if not defined(HAVE_MKL)
    fftwf_plan_with_nthreads(4);
    #endif

    // Execute FFT for all polarizations in parallel
    #pragma omp parallel for
	for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
		fftwf_complex *data = (fftwf_complex *) _data + pol * (size * size);

        // Create plan
        fftwf_plan plan;
        #pragma omp critical
        plan = fftwf_plan_dft_2d(size, size,
                                 data, data,
                                 sign, FFTW_ESTIMATE);

        // Execute FFTs
        fftwf_execute_dft(plan, data, data);

        // Scaling in case of an inverse FFT, so that FFT(iFFT())=identity()
        // => scale by 1/(N*N); since we only use half of the visibilities
        // scale real part by two, and set imaginary part to zero
        if (sign == FFTW_BACKWARD) {
            float scale_real = 2.0f / (float(size)*float(size));
            float scale_imag = 0.0f;
            for (int i = 0; i < size*size; i++) {
                data[i][0] *= scale_real;
                data[i][1] *= scale_imag;
            }
        }

        // Destroy plan
        #pragma omp critical
        fftwf_destroy_plan(plan);
    }
}

void kernel_fft_subgrid(
	long size,
	long batch,
    fftwf_complex *_data,
	int sign
	) {

    fftwf_complex *data = (fftwf_complex *) _data;

    // 2D FFT
    int rank = 2;

    // For grids of size*size elements
    int n[] = {(int) size, (int) size};

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
    plan = fftwf_plan_many_dft(
        rank, n, NR_POLARIZATIONS, _data, n,
        istride, idist, _data, n,
        ostride, odist, sign, flags);

    #pragma omp parallel for private(data)
    for (int i = 0; i < batch; i++) {
        data = (fftwf_complex *) _data + i * (NR_POLARIZATIONS * size * size);

        // Execute FFTs
        fftwf_execute_dft(plan, data, data);

        // Scaling in case of an inverse FFT, so that FFT(iFFT())=identity()
        if (sign == FFTW_BACKWARD) {
            float scale = 1 / (double(size)*double(size));
            for (int i = 0; i < NR_POLARIZATIONS*size*size; i++) {
                data[i][0] *= scale;
                data[i][1] *= scale;
            }
        }

    } // end for batch

    // Cleanup
    fftwf_destroy_plan(plan);
}

void kernel_fft(
    long gridsize,
	long size,
	long batch,
    fftwf_complex *data,
	int sign
	) {
    if (size == gridsize) {  // a bit of a hack; TODO: make separate functions for two cases
        kernel_fft_grid(size, data, sign);
	} else {
		kernel_fft_subgrid(size, batch, data, sign);
	}
}
}
