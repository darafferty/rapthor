#include <complex>

#include <math.h>
#include <stdio.h>
#include <immintrin.h>
#include <string.h>
#include <stdint.h>

#if defined(USING_INTEL_CXX_COMPILER)
#define VML_PRECISION VML_LA
#include <mkl_vml.h>
#endif

#include "Types.h"

extern "C" {
#if defined(USING_INTEL_CXX_COMPILER)
void kernel_degridder_intel(
    const int jobsize, const float w_offset,
	const UVWType		 __restrict__ *uvw,
	const WavenumberType __restrict__ *wavenumbers,
	VisibilitiesType	 __restrict__ *visibilities,
	const SpheroidalType __restrict__ *spheroidal,
	const ATermType		 __restrict__ *aterm,
	const MetadataType	 __restrict__ *metadata,
	const SubGridType	 __restrict__ *subgrid
    )
{
    // Load metadata
    const Metadata m = (*metadata)[0];
    const int baseline_offset_1 = m.baseline_offset;
    const int time_offset_1 = m.time_offset; // should be 0

    #pragma omp parallel shared(uvw, wavenumbers, visibilities, spheroidal, aterm, metadata)
    {
    // Iterate all subgrids
    #pragma omp for
	for (int s = 0; s < jobsize; s++) {
        // Load metadata
        const Metadata m = (*metadata)[s];
        const int time_nr = 0; // TODO: m.time_nr; will be aterm_index
        const int offset = (m.baseline_offset - baseline_offset_1)
                           + (m.time_offset - time_offset_1);
        const int nr_timesteps = m.nr_timesteps;
        const int station1 = m.baseline.station1;
        const int station2 = m.baseline.station2;
        const int x_coordinate = m.coordinate.x;
        const int y_coordinate = m.coordinate.y;

        // Storage for precomputed values
        FLOAT_COMPLEX _pixels[SUBGRIDSIZE][SUBGRIDSIZE][NR_POLARIZATIONS] __attribute__((aligned(32)));
        float pixels_real[NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));
        float pixels_imag[NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));
        float phase_index[SUBGRIDSIZE][SUBGRIDSIZE]  __attribute__((aligned(32)));
        float phase_offset[SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));
        float phasor_real[SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));
        float phasor_imag[SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));
        float phase[SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));

        // Compute u and v offset in wavelenghts
        const float u_offset = (x_coordinate + SUBGRIDSIZE/2) / IMAGESIZE;
        const float v_offset = (y_coordinate + SUBGRIDSIZE/2) / IMAGESIZE;

        // Apply aterm to subgrid
        for (int y = 0; y < SUBGRIDSIZE; y++) {
            for (int x = 0; x < SUBGRIDSIZE; x++) {
                // Load aterm for station1
                FLOAT_COMPLEX aXX1 = (*aterm)[station1][time_nr][0][y][x];
                FLOAT_COMPLEX aXY1 = (*aterm)[station1][time_nr][1][y][x];
                FLOAT_COMPLEX aYX1 = (*aterm)[station1][time_nr][2][y][x];
                FLOAT_COMPLEX aYY1 = (*aterm)[station1][time_nr][3][y][x];

                // Load aterm for station2
                FLOAT_COMPLEX aXX2 = conj((*aterm)[station2][time_nr][0][y][x]);
                FLOAT_COMPLEX aXY2 = conj((*aterm)[station2][time_nr][1][y][x]);
                FLOAT_COMPLEX aYX2 = conj((*aterm)[station2][time_nr][2][y][x]);
                FLOAT_COMPLEX aYY2 = conj((*aterm)[station2][time_nr][3][y][x]);

                // Load spheroidal
                float _spheroidal = (*spheroidal)[y][x];

                // Compute shifted position in subgrid
                int x_src = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                int y_src = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

                // Load uv values
                FLOAT_COMPLEX pixelsXX = _spheroidal * (*subgrid)[s][0][y_src][x_src];
                FLOAT_COMPLEX pixelsXY = _spheroidal * (*subgrid)[s][1][y_src][x_src];
                FLOAT_COMPLEX pixelsYX = _spheroidal * (*subgrid)[s][2][y_src][x_src];
                FLOAT_COMPLEX pixelsYY = _spheroidal * (*subgrid)[s][3][y_src][x_src];

                // Apply aterm to subgrid
                _pixels[y][x][0]  = pixelsXX * aXX1;
                _pixels[y][x][0] += pixelsXY * aYX1;
                _pixels[y][x][0] += pixelsXX * aXX2;
                _pixels[y][x][0] += pixelsYX * aYX2;
                _pixels[y][x][1]  = pixelsXX * aXY1;
                _pixels[y][x][1] += pixelsXY * aYY1;
                _pixels[y][x][1] += pixelsXY * aXX2;
                _pixels[y][x][1] += pixelsYY * aYX2;
                _pixels[y][x][2]  = pixelsYX * aXX1;
                _pixels[y][x][2] += pixelsYY * aYX1;
                _pixels[y][x][2] += pixelsXX * aXY2;
                _pixels[y][x][2] += pixelsYX * aYY2;
                _pixels[y][x][3]  = pixelsYX * aXY1;
                _pixels[y][x][3] += pixelsYY * aYY1;
                _pixels[y][x][3] += pixelsXY * aXY2;
                _pixels[y][x][3] += pixelsYY * aYY2;
            }
        }

        // Copy pixels from _pixels to pixels_real and pixels_imag
        for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
            for (int y = 0; y < SUBGRIDSIZE; y++) {
                for (int x = 0; x < SUBGRIDSIZE; x++) {
                    pixels_real[pol][y][x] =  _pixels[y][x][pol].real();
                    pixels_imag[pol][y][x] =  _pixels[y][x][pol].imag();
                }
            }
        }

        // Reset all visibilities
        // TODO: is this memset necessary?
        // memset(&(*visibilities)[offset][0][0], 0, nr_timesteps * NR_CHANNELS * NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX));

        // Iterate all timesteps
        for (int time = 0; time < nr_timesteps; time++) {
            // Load UVW coordinates
            float u = (*uvw)[offset + time].u;
            float v = (*uvw)[offset + time].v;
            float w = (*uvw)[offset + time].w;

            // Compute phase indices and phase offsets
            for (int y = 0; y < SUBGRIDSIZE; y++) {
                for (int x = 0; x < SUBGRIDSIZE; x++) {
                    // Compute l,m,n
                    float l = -(x-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
                    float m =  (y-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
                    float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));

                    // Compute phase index
                    phase_index[y][x] = u*l + v*m + w*n;

                    // Compute phase offset
                    phase_offset[y][x] = u_offset*l + v_offset*m + w_offset*n;
                }
            }

            // Iterate all channels
            for (int chan = 0; chan < NR_CHANNELS; chan++) {
                // Compute phase
                for (int y = 0; y < SUBGRIDSIZE; y++) {
                    for (int x = 0; x < SUBGRIDSIZE; x++) {
                        phase[y][x] = (phase_index[y][x] * (*wavenumbers)[chan]) - phase_offset[y][x];
                    }
                }

                // Compute phasor
                vmsSinCos(SUBGRIDSIZE * SUBGRIDSIZE,
                    (const float *) &phase[0][0],
                                    &phasor_imag[0][0],
                                    &phasor_real[0][0], VML_PRECISION);

                // Storage for sums
                float sum_xx_real, sum_xx_imag;
                float sum_xy_real, sum_xy_imag;
                float sum_yx_real, sum_yx_imag;
                float sum_yy_real, sum_yy_imag;

                // Accumulate visibilities
                for (int y = 0; y < SUBGRIDSIZE; y++) {
                    for (int x = 0; x < SUBGRIDSIZE; x++) {
                        float _phasor_real = phasor_real[y][x];
                        float _phasor_imag = phasor_imag[y][x];

                        sum_xx_real += _phasor_real * pixels_real[0][y][x];
                        sum_xx_imag += _phasor_real * pixels_imag[0][y][x];
                        sum_xx_real -= _phasor_imag * pixels_imag[0][y][x];
                        sum_xx_imag += _phasor_imag * pixels_real[0][y][x];

                        sum_xy_real += _phasor_real * pixels_real[1][y][x];
                        sum_xy_imag += _phasor_real * pixels_imag[1][y][x];
                        sum_xy_real -= _phasor_imag * pixels_imag[1][y][x];
                        sum_xy_imag += _phasor_imag * pixels_real[1][y][x];

                        sum_yx_real += _phasor_real * pixels_real[2][y][x];
                        sum_yx_imag += _phasor_real * pixels_imag[2][y][x];
                        sum_yx_real -= _phasor_imag * pixels_imag[2][y][x];
                        sum_yx_imag += _phasor_imag * pixels_real[2][y][x];

                        sum_yy_real += _phasor_real * pixels_real[3][y][x];
                        sum_yy_imag += _phasor_real * pixels_imag[3][y][x];
                        sum_yy_real -= _phasor_imag * pixels_imag[3][y][x];
                        sum_yy_imag += _phasor_imag * pixels_real[3][y][x];
                    }
                }

                // Store visibilities
                (*visibilities)[offset + time][chan][0] = {sum_xx_real, sum_xx_imag};
                (*visibilities)[offset + time][chan][1] = {sum_xy_real, sum_xy_imag};
                (*visibilities)[offset + time][chan][2] = {sum_yx_real, sum_yx_imag};
                (*visibilities)[offset + time][chan][3] = {sum_yy_real, sum_yy_imag};
            }
        }
	}
    }
}
#endif

#if defined(USING_GNU_CXX_COMPILER)
void kernel_degridder_gnu(
	const int jobsize, const float w_offset,
	const UVWType		   __restrict__ *uvw,
	const WavenumberType   __restrict__ *wavenumbers,
	VisibilitiesType       __restrict__ *visibilities,
	const SpheroidalType   __restrict__ *spheroidal,
	const ATermType		   __restrict__ *aterm,
	const MetadataType	   __restrict__ *metadata,
	const SubGridType	   __restrict__ *subgrid
	) {
    printf("%s not implemented yet\n", __func__);
}
#endif

void kernel_degridder(
	const int jobsize, const float w_offset,
	const UVWType		   __restrict__ *uvw,
	const WavenumberType   __restrict__ *wavenumbers,
	VisibilitiesType       __restrict__ *visibilities,
	const SpheroidalType   __restrict__ *spheroidal,
	const ATermType		   __restrict__ *aterm,
	const MetadataType	   __restrict__ *metadata,
	const SubGridType	   __restrict__ *subgrid
	) {
    #if defined(USING_INTEL_CXX_COMPILER)
    kernel_degridder_intel(
          jobsize, w_offset, uvw, wavenumbers,
          visibilities, spheroidal, aterm, metadata, subgrid);
    #elif defined(USING_GNU_CXX_COMPILER)
    kernel_degridder_gnu(
          jobsize, w_offset, uvw, wavenumbers,
          visibilities, spheroidal, aterm, metadata, subgrid);
    #else
    printf("%s not implemented yet, use Intel or GNU compiler\n", __func__);
    #endif
}
}
