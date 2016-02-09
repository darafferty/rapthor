#include <complex>

#include <math.h>
#include <stdio.h>
#include <immintrin.h>
#include <string.h>
#include <stdint.h>

#if defined(__INTEL_COMPILER)
#define USE_VML
#define VML_PRECISION VML_LA
#include <mkl_vml.h>
#endif

#include "Types.h"

extern "C" {
void kernel_degridder(
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
    // Find offset of first subgrid
    const Metadata m = (*metadata)[0];
    const int baseline_offset_1 = m.baseline_offset;
    const int time_offset_1     = m.time_offset; // should be 0

    #pragma omp parallel shared(uvw, wavenumbers, visibilities, spheroidal, aterm, metadata)
    {
        // Iterate all subgrids
        #pragma omp for
        for (int s = 0; s < jobsize; s++) {

            // Load metadata
            const Metadata m = (*metadata)[s];
            const int offset = (m.baseline_offset - baseline_offset_1)
                  + (m.time_offset - time_offset_1);
            const int nr_timesteps = m.nr_timesteps;
            const int aterm_index = m.aterm_index;
            const int station1 = m.baseline.station1;
            const int station2 = m.baseline.station2;
            const int x_coordinate = m.coordinate.x;
            const int y_coordinate = m.coordinate.y;

            // Storage
            FLOAT_COMPLEX pixels[NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));
            float pixels_real[NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));
            float pixels_imag[NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));

            // Apply aterm to subgrid
            for (int y = 0; y < SUBGRIDSIZE; y++) {
                for (int x = 0; x < SUBGRIDSIZE; x++) {
                    // Load aterm for station1
                    FLOAT_COMPLEX aXX1 = (*aterm)[station1][aterm_index][0][y][x];
                    FLOAT_COMPLEX aXY1 = (*aterm)[station1][aterm_index][1][y][x];
                    FLOAT_COMPLEX aYX1 = (*aterm)[station1][aterm_index][2][y][x];
                    FLOAT_COMPLEX aYY1 = (*aterm)[station1][aterm_index][3][y][x];

                    // Load aterm for station2
                    FLOAT_COMPLEX aXX2 = conj((*aterm)[station2][aterm_index][0][y][x]);
                    FLOAT_COMPLEX aXY2 = conj((*aterm)[station2][aterm_index][1][y][x]);
                    FLOAT_COMPLEX aYX2 = conj((*aterm)[station2][aterm_index][2][y][x]);
                    FLOAT_COMPLEX aYY2 = conj((*aterm)[station2][aterm_index][3][y][x]);

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
                    pixels[0][y][x]  = pixelsXX * aXX1;
                    pixels[0][y][x] += pixelsXY * aYX1;
                    pixels[1][y][x]  = pixelsXX * aXY1;
                    pixels[1][y][x] += pixelsXY * aYY1;
                    pixels[2][y][x]  = pixelsYX * aXX1;
                    pixels[2][y][x] += pixelsYY * aYX1;
                    pixels[3][y][x]  = pixelsYX * aXY1;
                    pixels[3][y][x] += pixelsYY * aYY1;

                    pixelsXX = pixels[0][y][x];
                    pixelsXY = pixels[1][y][x];
                    pixelsYX = pixels[2][y][x];
                    pixelsYY = pixels[3][y][x];
                    pixels[0][y][x]  = pixelsXX * aXX2;
                    pixels[0][y][x] += pixelsYX * aYX2;
                    pixels[1][y][x]  = pixelsXY * aXX2;
                    pixels[1][y][x] += pixelsYY * aYX2;
                    pixels[2][y][x]  = pixelsXX * aXY2;
                    pixels[2][y][x] += pixelsYX * aYY2;
                    pixels[3][y][x]  = pixelsXY * aXY2;
                    pixels[3][y][x] += pixelsYY * aYY2;
                } // end x
            } // end y

            // Split real and imaginary part of pixels
            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                for (int y = 0; y < SUBGRIDSIZE; y++) {
                    for (int x = 0; x < SUBGRIDSIZE; x++) {
                        pixels_real[pol][y][x] = pixels[pol][y][x].real();
                        pixels_imag[pol][y][x] = pixels[pol][y][x].imag();
                    } // end x
                } // end y
            }

            // Compute u and v offset in wavelenghts
            const float u_offset = (x_coordinate + SUBGRIDSIZE/2 - GRIDSIZE/2)
                                   * (2*M_PI / IMAGESIZE);
            const float v_offset = (y_coordinate + SUBGRIDSIZE/2 - GRIDSIZE/2)
                                   * (2*M_PI / IMAGESIZE);

            // Iterate all timesteps
            for (int time = 0; time < nr_timesteps; time++) {
                // Load UVW coordinates
                float u = (*uvw)[offset + time].u;
                float v = (*uvw)[offset + time].v;
                float w = (*uvw)[offset + time].w;

                float phase_index[SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));
                float phase_offset[SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));

                for (int y = 0; y < SUBGRIDSIZE; y++) {
                    for (int x = 0; x < SUBGRIDSIZE; x++) {
                        // Compute l,m,n
                        const float l = (x-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
                        const float m = (y-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
                        // evaluate n = 1.0f - sqrt(1.0 - (l * l) - (m * m));
                        // accurately for small values of l and m
                        const float tmp = (l * l) + (m * m);
                        const float n = tmp / (1.0f + sqrtf(1.0f - tmp));

                        // Compute phase index
                        phase_index[y][x] = u*l + v*m + w*n;

                        // Compute phase offset
                        phase_offset[y][x] = u_offset*l + v_offset*m + w_offset*n;
                    }
                }

                // Iterate all channels
                for (int chan = 0; chan < NR_CHANNELS; chan++) {

                    float phase[SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));
                    float phasor_imag[SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));
                    float phasor_real[SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));

                    #pragma nofusion
                    for (int y = 0; y < SUBGRIDSIZE; y++) {
                        for (int x = 0; x < SUBGRIDSIZE; x++) {
                            // Compute phase
                            float wavenumber = (*wavenumbers)[chan];
                            phase[y][x] = phase_offset[y][x] - (phase_index[y][x] * wavenumber);
                        }
                    }

                    #if defined(USE_VML)
                    vmsSinCos(SUBGRIDSIZE * SUBGRIDSIZE,
                              (const float *) &phase[0][0],
                              &phasor_imag[0][0],
                              &phasor_real[0][0], VML_PRECISION);
                    #else
                    #pragma nofusion
                    for (int y = 0; y < SUBGRIDSIZE; y++) {
                        for (int x = 0; x < SUBGRIDSIZE; x++) {
                            // Compute phasor
                            phasor_imag[y][x] = sinf(phase[y][x]);
                            phasor_real[y][x] = cosf(phase[y][x]);
                        }
                    }
                    #endif

                    // Storage for sums
                    float sum_xx_real = 0.0f, sum_xx_imag = 0.0f;
                    float sum_xy_real = 0.0f, sum_xy_imag = 0.0f;
                    float sum_yx_real = 0.0f, sum_yx_imag = 0.0f;
                    float sum_yy_real = 0.0f, sum_yy_imag = 0.0f;

                    #pragma nofusion
                    for (int y = 0; y < SUBGRIDSIZE; y++) {
                        #pragma omp simd reduction(+:\
                            sum_xx_real, sum_xx_imag,\
                            sum_xy_real, sum_xy_imag,\
                            sum_yx_real, sum_yx_imag,\
                            sum_yy_real, sum_yy_imag)
                        for (int x = 0; x < SUBGRIDSIZE; x++) {
                            sum_xx_real +=  phasor_real[y][x] * pixels_real[0][y][x];
                            sum_xx_imag +=  phasor_real[y][x] * pixels_imag[0][y][x];
                            sum_xx_real += -phasor_imag[y][x] * pixels_imag[0][y][x];
                            sum_xx_imag +=  phasor_imag[y][x] * pixels_real[0][y][x];

                            sum_xy_real +=  phasor_real[y][x] * pixels_real[1][y][x];
                            sum_xy_imag +=  phasor_real[y][x] * pixels_imag[1][y][x];
                            sum_xy_real += -phasor_imag[y][x] * pixels_imag[1][y][x];
                            sum_xy_imag +=  phasor_imag[y][x] * pixels_real[1][y][x];

                            sum_yx_real +=  phasor_real[y][x] * pixels_real[2][y][x];
                            sum_yx_imag +=  phasor_real[y][x] * pixels_imag[2][y][x];
                            sum_yx_real += -phasor_imag[y][x] * pixels_imag[2][y][x];
                            sum_yx_imag +=  phasor_imag[y][x] * pixels_real[2][y][x];

                            sum_yy_real +=  phasor_real[y][x] * pixels_real[3][y][x];
                            sum_yy_imag +=  phasor_real[y][x] * pixels_imag[3][y][x];
                            sum_yy_real += -phasor_imag[y][x] * pixels_imag[3][y][x];
                            sum_yy_imag +=  phasor_imag[y][x] * pixels_real[3][y][x];
                        }
                    }

                    // Store visibilities
                    const float scale = 1.0f / (SUBGRIDSIZE*SUBGRIDSIZE);
                    (*visibilities)[offset + time][chan][0] = {scale*sum_xx_real, scale*sum_xx_imag};
                    (*visibilities)[offset + time][chan][1] = {scale*sum_xy_real, scale*sum_xy_imag};
                    (*visibilities)[offset + time][chan][2] = {scale*sum_yx_real, scale*sum_yx_imag};
                    (*visibilities)[offset + time][chan][3] = {scale*sum_yy_real, scale*sum_yy_imag};
                } // end for channel
            } // end for time
        } // end for s
    } // end #pragma parallel
}
} // end extern "C"
