#pragma omp declare target

#include <complex>
#include <cmath>
#include <fftw3.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "Types.h"

namespace idg {
namespace kernel {
namespace knc {

void fft(
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
            float scale_real = 1 / (float(size)*float(size));
            float scale_imag = 1 / (float(size)*float(size));
            // TODO: A bit of a hack to have it here:
            // Since we only take half the visibilities
            // scale real part by two, and set imaginery part to zero
            // Since gridsize is not know at this point, batch is used
            // to detect fft operation on a grid, instead of on subgrids
            if (batch == 1) {
                scale_real *= 2;
                scale_imag *= 0;
            }
            for (int i = 0; i < nr_polarizations*size*size; i++) {
                data[i][0] *= scale_real;
                data[i][1] *= scale_imag;
            }
        }

        // Destroy plan
        fftwf_destroy_plan(plan);
    }
}

void ifftshift(std::complex<float> *array)
{
    // TODO: implement
}

void fftshift(int gridsize, std::complex<float> *array)
{
    auto buffer = (std::complex<float> *) malloc(gridsize * sizeof(std::complex<float>));

    if (gridsize % 2 != 0)
        throw std::invalid_argument("gridsize is assumed to be even");

    for (int i = 0; i < gridsize/2; i++) {
        // save i-th row into buffer
        memcpy(buffer, &array[i*gridsize],
               gridsize*sizeof(std::complex<float>));

        auto j = i + gridsize/2;
        memcpy(&array[i*gridsize + gridsize/2], &array[j*gridsize],
               (gridsize/2)*sizeof(std::complex<float>));
        memcpy(&array[i*gridsize], &array[j*gridsize + gridsize/2],
               (gridsize/2)*sizeof(std::complex<float>));
        memcpy(&array[j*gridsize], &buffer[gridsize/2],
               (gridsize/2)*sizeof(std::complex<float>));
        memcpy(&array[j*gridsize + gridsize/2], &buffer[0],
               (gridsize/2)*sizeof(std::complex<float>));
    }

    //delete [] buffer;
    free(buffer);
}

void fftshift(int nr_polarizations, int gridsize, std::complex<float> *grid)
{
    #pragma omp parallel for
    for (int p = 0; p < nr_polarizations; p++) {
        // Pass &grid[p][GRIDSIZE][GRIDSIZE]
        // and do shift for each polarization
        fftshift(gridsize, grid + p*gridsize*gridsize);
    }
}

void ifftshift(int nr_polarizations, int gridsize, std::complex<float> *grid)
{
    // TODO: implement for odd size gridsize
    // For even gridsize, same as fftshift
    fftshift(nr_polarizations, gridsize, grid);
}

} // end namespace knc
} // end namespace kernel
} // end namespace idg

#pragma omp end declare target
