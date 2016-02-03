#include <complex>

#include <math.h>
#include <fftw3.h>
#include <stdint.h>

#include "Types.h"


extern "C" {
void kernel_fft_grid(
	int size, 
	fftwf_complex __restrict__ *_data,
    int sign    // -1=FFTW_FORWARD, 1=FFTW_BACKWARD
	) {
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
            #pragma omp parallel for
            for (int i = 0; i < size*size; i++) {
                data[i][0] *= scale_real;
                data[i][1] *= scale_imag;
            }
        }
        
        // Destroy plan
        fftwf_destroy_plan(plan);
    }
}

void kernel_fft_subgrid(
	int size, 
	int batch,
	fftwf_complex __restrict__ *_data,
	int sign
	) {
    #pragma omp parallel for
    for (int i = 0; i < batch; i++) {
        fftwf_complex *data = (fftwf_complex *) _data + i * (NR_POLARIZATIONS * size * size);	
        
        // 2D FFT
        int rank = 2;
        
        // For grids of size*size elements
        int n[] = {size, size};
        
        // Set stride
        int istride = 1;
        int ostride = istride;
        
        // Set dist
        int idist = n[0] * n[1];
        int odist = idist;
        
        // Planner flags
        int flags = FFTW_ESTIMATE;
        fftwf_plan plan;
        #pragma omp critical
        plan = fftwf_plan_many_dft(
            rank, n, NR_POLARIZATIONS, data, n,
            istride, idist, data, n,
            ostride, odist, sign, flags);
        
        // Execute FFTs
        fftwf_execute_dft(plan, data, data);

        // Scaling in case of an inverse FFT, so that FFT(iFFT())=identity()
        if (sign == FFTW_BACKWARD) {
            float scale = 1 / (double(size)*double(size));
            #pragma omp parallel for
            for (int i = 0; i < NR_POLARIZATIONS*size*size; i++) {
                data[i][0] *= scale;
                data[i][1] *= scale;
            }
        }
        
        // Destroy plan
        fftwf_destroy_plan(plan);
    }
}

void kernel_fft(
	int size, 
	int batch,
	fftwf_complex __restrict__ *data,
	int sign
	) {
    if (batch == 1) {
		kernel_fft_grid(size, data, sign);
	} else {
		kernel_fft_subgrid(size, batch, data, sign);
	}
}
}
