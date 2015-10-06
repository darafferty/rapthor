#include <complex>

#include <math.h>
#include <fftw3.h>
#include <stdint.h>

#include "Types.h"

#define FFT_LAYOUT_YXP (-1)
#define FFT_LAYOUT_PYX (+1)


extern "C" {
void kernel_fft_grid(
	int size, 
	fftwf_complex __restrict__ *_data,
	int sign
	) {
        #pragma omp parallel for
	for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
		fftwf_complex __restrict__ *data = (fftwf_complex *) _data + pol * (size * size);
	
	        // Create plan
		fftwf_plan plan = fftwf_plan_dft_2d(size, size,
	            		    data, data,
	            		    sign, FFTW_ESTIMATE);
	        // Execute FFTs
	        fftwf_execute_dft(plan, data, data);
	        
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
printf("fft subgrid, batch=%d\n", batch);
		#pragma omp parallel for
		for (int i = 0; i < batch; i++) {
			fftwf_complex __restrict__ *data = (fftwf_complex *) _data + i * (NR_POLARIZATIONS * size * size);	

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
            		fftwf_plan plan = fftwf_plan_many_dft(
            		    rank, n, NR_POLARIZATIONS, data, n,
            		    istride, idist, data, n,
            		    ostride, odist, sign, flags);

            		// Execute FFTs
            		fftwf_execute_dft(plan, data, data);
            		
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
 
uint64_t kernel_fft_flops(int size, int batch) {
	return 1ULL * batch * NR_POLARIZATIONS * 5 * size * size * log(size * size);
}

uint64_t kernel_fft_bytes(int size, int batch) {
	return 1ULL * 2 * batch * NR_POLARIZATIONS * size * size * sizeof(FLOAT_COMPLEX);
}
}
