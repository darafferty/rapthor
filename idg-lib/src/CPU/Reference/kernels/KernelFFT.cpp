#include <complex>

#include <math.h>
#include <fftw3.h>
#include <stdint.h>

#include "Types.h"

extern "C" {
void kernel_fft(
    long gridsize,
	long size,
	long batch,
    fftwf_complex *data,
    int sign    // Note: -1=FFTW_FORWARD, 1=FFTW_BACKWARD
	) {

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
    #pragma omp critical
    {
            plan = fftwf_plan_many_dft(
            rank, n, batch * NR_POLARIZATIONS, data, n,
            istride, idist, data, n,
            ostride, odist, sign, flags);
    }

    // Execute FFTs
    fftwf_execute_dft(plan, data, data);

    // Scaling in case of an inverse FFT, so that FFT(iFFT())=identity()
    if (sign == FFTW_BACKWARD) {

        float scale_real = 1.0f / (float(size)*float(size));
        float scale_imag = 1.0f / (float(size)*float(size));

        // TODO: A bit of a hack to have it here:
        // Since we only take half the visibilities
        // scale real part by two, and set imaginery part to zero
        if (size == gridsize) {
            scale_real = 2.0f / (float(size)*float(size));
            scale_imag = 0.0f;
        }

        #pragma omp parallel for
        for (int i = 0; i < batch*NR_POLARIZATIONS*size*size; i++) {
            data[i][0] *= scale_real;
            data[i][1] *= scale_imag;
        }

    }

    // Destroy plan
    #pragma omp critical
    fftwf_destroy_plan(plan);
}
}
