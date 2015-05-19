#include <math.h>
#include <fftw3.h>

#include <idg/Common/Types.h>

extern "C" {

/*
	Kernel
*/
#if ORDER == ORDER_BL_V_U_P
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
	int istride = NR_POLARIZATIONS;
	int ostride = istride;

    // Set dist	
	int idist = n[0] * n[1] * NR_POLARIZATIONS;
	int odist = idist;

	// Planner flags
	int flags = FFTW_ESTIMATE;
	
	// Create plan
	fftwf_plan plan = fftwf_plan_many_dft(rank, n, batch, data, n,
										  istride, idist, data, n,
										  ostride, odist, sign, flags);

	// Execute FFTs
	fftwf_complex *data_xx = data + 0;
	fftwf_complex *data_xy = data + 1;
	fftwf_complex *data_yx = data + 2;
	fftwf_complex *data_yy = data + 3;
	fftwf_execute_dft(plan, data_xx, data_xx);
	fftwf_execute_dft(plan, data_xy, data_xy);
	fftwf_execute_dft(plan, data_yx, data_yx);
	fftwf_execute_dft(plan, data_yy, data_yy);
	
	// Destroy plan
	fftwf_destroy_plan(plan);
}
#elif ORDER == ORDER_BL_P_V_U
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
	fftwf_plan plan = fftwf_plan_many_dft(rank, n, batch, data, n,
										  istride, idist, data, n,
										  ostride, odist, sign, flags);

	// Execute FFTs
	fftwf_execute_dft(plan, data, data);
	
	// Destroy plan
	fftwf_destroy_plan(plan);
}
#endif

#include <idg/Common/Parameters.h>

}
