#pragma omp declare target

#include <complex>
#include <cmath>
#include <fftw3.h>
#include <cstdint>

#include "Types.h"

namespace idg {
namespace kernel {
namespace knc {

void kernel_fft(
	int size,
	int batch,
	void *_data,
	int sign,
    int nr_polarizations)
{
    #pragma omp parallel for
    for (int i = 0; i < batch; i++) {
        fftwf_complex __restrict__ *data = (fftwf_complex *) _data + i * (nr_polarizations * size * size);

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
            rank, n, nr_polarizations, data, n,
            istride, idist, data, n,
            ostride, odist, sign, flags);

        // Execute FFTs
        fftwf_execute_dft(plan, data, data);

        // Scaling in case of an inverse FFT, so that FFT(iFFT())=identity()
        if (sign == FFTW_BACKWARD) {
            float scale = 1 / (float(size)*float(size));
            for (int i = 0; i < nr_polarizations*size*size; i++) {
                data[i][0] *= scale;
                data[i][1] *= scale;
            }
        }

        // Destroy plan
        fftwf_destroy_plan(plan);
    }
}

} // end namespace knc
} // end namespace kernel
} // end namespace idg

#pragma omp end declare target
