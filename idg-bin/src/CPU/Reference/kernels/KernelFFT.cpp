#include <complex>

#include <math.h>
#include <fftw3.h>
#include <stdint.h>

#include "Types.h"

#define FFT_LAYOUT_YXP (-1)
#define FFT_LAYOUT_PYX (+1)


extern "C" {
void kernel_fft(
	int size, 
	int batch,
	fftwf_complex __restrict__ *data,
	int sign
	) {
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
    
    // Create plan
    fftwf_plan plan = fftwf_plan_many_dft(
        rank, n, batch * NR_POLARIZATIONS, data, n,
        istride, idist, data, n,
        ostride, odist, sign, flags);

    // Execute FFTs
    fftwf_execute_dft(plan, data, data);
    
    // Destroy plan
    fftwf_destroy_plan(plan);
}
}
