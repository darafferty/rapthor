#include <complex>

#include <math.h>
#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
#include <string.h>
#include <stdint.h>

#if defined(__INTEL_COMPILER)
#define USE_VML
#define VML_PRECISION VML_LA
#include <mkl_vml.h>
#endif

#include "Types.h"

extern "C" {
void kernel_gridder(
    const int jobsize, const float w_offset,
    const UVWType		   __restrict__ *uvw,
    const WavenumberType   __restrict__ *wavenumbers,
    const VisibilitiesType __restrict__ *visibilities,
    const SpheroidalType   __restrict__ *spheroidal,
    const ATermType		   __restrict__ *aterm,
    const MetadataType	   __restrict__ *metadata,
    SubGridType			   __restrict__ *subgrid
    )
{
    // Find offset of first subgrid
    const Metadata m = (*metadata)[0];
    const int baseline_offset_1 = m.baseline_offset;
    const int time_offset_1 = m.time_offset; // should be 0

    // Iterate all subgrids
    #pragma omp parallel shared(uvw, wavenumbers, visibilities, spheroidal, aterm, metadata)
    {
        // Iterate all subgrids
        #pragma omp for // schedule(dynamic)
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

            // Compute u and v offset in wavelenghts
            const float u_offset = (x_coordinate + SUBGRIDSIZE/2 - GRIDSIZE/2)
                * (2*M_PI / IMAGESIZE);
            const float v_offset = (y_coordinate + SUBGRIDSIZE/2 - GRIDSIZE/2)
                * (2*M_PI / IMAGESIZE);

            float vis_real[nr_timesteps][NR_POLARIZATIONS][NR_CHANNELS] __attribute__((aligned(32)));
            float vis_imag[nr_timesteps][NR_POLARIZATIONS][NR_CHANNELS] __attribute__((aligned(32)));
            for (int time = 0; time < nr_timesteps; time++) {
                for (int chan = 0; chan < NR_CHANNELS; chan++) {
                    for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        vis_real[time][pol][chan] = (*visibilities)[offset + time][chan][pol].real();
                        vis_imag[time][pol][chan] = (*visibilities)[offset + time][chan][pol].imag();
                    }
                }
            }

            // Iterate all pixels in subgrid
            for (int y = 0; y < SUBGRIDSIZE; y++) {
                for (int x = 0; x < SUBGRIDSIZE; x++) {
                    // Initialize pixel for every polarization
                    float pixels_xx_real = 0.0f;
                    float pixels_xy_real = 0.0f;
                    float pixels_yx_real = 0.0f;
                    float pixels_yy_real = 0.0f;
                    float pixels_xx_imag = 0.0f;
                    float pixels_xy_imag = 0.0f;
                    float pixels_yx_imag = 0.0f;
                    float pixels_yy_imag = 0.0f;

                    // Compute l,m,n
                    const float l = (x-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
                    const float m = (y-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
                    // evaluate n = 1.0f - sqrt(1.0 - (l * l) - (m * m));
                    // accurately for small values of l and m
                    const float tmp = (l * l) + (m * m);
                    const float n = tmp / (1.0f + sqrtf(1.0f - tmp));

                    float phase[nr_timesteps][NR_CHANNELS] __attribute__((aligned(32)));
                    float phasor_real[nr_timesteps][NR_CHANNELS] __attribute__((aligned(32)));
                    float phasor_imag[nr_timesteps][NR_CHANNELS] __attribute__((aligned(32)));

                    // Iterate all timesteps
                    #pragma nofusion
                    for (int time = 0; time < nr_timesteps; time++) {
                        // Load UVW coordinates
                        float u = (*uvw)[offset + time].u;
                        float v = (*uvw)[offset + time].v;
                        float w = (*uvw)[offset + time].w;

                        // Compute phase index
                        float phase_index = u*l + v*m + w*n;

                        // Compute phase offset
                        float phase_offset = u_offset*l + v_offset*m + w_offset*n;

                        // #pragma nofusion
                        for (int chan = 0; chan < NR_CHANNELS; chan++) {
                            // Compute phase
                            float wavenumber = (*wavenumbers)[chan];
                            phase[time][chan]  = (phase_index * wavenumber) - phase_offset;
                        }
                    } // end time

                    // Compute phasor
                    vmsSinCos(nr_timesteps * NR_CHANNELS,
                              &phase[0][0],
                              &phasor_imag[0][0],
                              &phasor_real[0][0], VML_PRECISION);

                    for (int time = 0; time < nr_timesteps; time++) {
                        // Update pixel for every channel
                        #pragma omp simd reduction(+:pixels_xx_real,pixels_xx_imag,pixels_xy_real,pixels_xy_imag,pixels_yx_real,pixels_yx_imag,pixels_yy_real,pixels_yy_imag)
                        for (int chan = 0; chan < NR_CHANNELS; chan++) {
                            // Update pixels
                            pixels_xx_real +=  vis_real[time][0][chan] * phasor_real[time][chan];
                            pixels_xx_imag +=  vis_real[time][0][chan] * phasor_imag[time][chan];
                            pixels_xx_real += -vis_imag[time][0][chan] * phasor_imag[time][chan];
                            pixels_xx_imag +=  vis_imag[time][0][chan] * phasor_real[time][chan];

                            pixels_xy_real +=  vis_real[time][1][chan] * phasor_real[time][chan];
                            pixels_xy_imag +=  vis_real[time][1][chan] * phasor_imag[time][chan];
                            pixels_xy_real += -vis_imag[time][1][chan] * phasor_imag[time][chan];
                            pixels_xy_imag +=  vis_imag[time][1][chan] * phasor_real[time][chan];

                            // #pragma distribute_point

                            pixels_yx_real +=  vis_real[time][2][chan] * phasor_real[time][chan];
                            pixels_yx_imag +=  vis_real[time][2][chan] * phasor_imag[time][chan];
                            pixels_yx_real += -vis_imag[time][2][chan] * phasor_imag[time][chan];
                            pixels_yx_imag +=  vis_imag[time][2][chan] * phasor_real[time][chan];

                            pixels_yy_real +=  vis_real[time][3][chan] * phasor_real[time][chan];
                            pixels_yy_imag +=  vis_real[time][3][chan] * phasor_imag[time][chan];
                            pixels_yy_real += -vis_imag[time][3][chan] * phasor_imag[time][chan];
                            pixels_yy_imag +=  vis_imag[time][3][chan] * phasor_real[time][chan];
                        }
                    }

                    FLOAT_COMPLEX pixels[NR_POLARIZATIONS];
                    pixels[0] = FLOAT_COMPLEX(pixels_xx_real, pixels_xx_imag);
                    pixels[1] = FLOAT_COMPLEX(pixels_xy_real, pixels_xy_imag);
                    pixels[2] = FLOAT_COMPLEX(pixels_yx_real, pixels_yx_imag);
                    pixels[3] = FLOAT_COMPLEX(pixels_yy_real, pixels_yy_imag);

                    // Load a term for station1
                    FLOAT_COMPLEX aXX1 = (*aterm)[station1][aterm_index][0][y][x];
                    FLOAT_COMPLEX aXY1 = (*aterm)[station1][aterm_index][1][y][x];
                    FLOAT_COMPLEX aYX1 = (*aterm)[station1][aterm_index][2][y][x];
                    FLOAT_COMPLEX aYY1 = (*aterm)[station1][aterm_index][3][y][x];

                    // Load aterm for station2
                    FLOAT_COMPLEX aXX2 = conj((*aterm)[station2][aterm_index][0][y][x]);
                    FLOAT_COMPLEX aXY2 = conj((*aterm)[station2][aterm_index][1][y][x]);
                    FLOAT_COMPLEX aYX2 = conj((*aterm)[station2][aterm_index][2][y][x]);
                    FLOAT_COMPLEX aYY2 = conj((*aterm)[station2][aterm_index][3][y][x]);

                    // Apply aterm to subgrid: P*A1
                    // [ pixels[0], pixels[1];    [ aXX1, aXY1;
                    //   pixels[2], pixels[3] ] *   aYX1, aYY1 ]
                    FLOAT_COMPLEX pixelsXX = pixels[0];
                    FLOAT_COMPLEX pixelsXY = pixels[1];
                    FLOAT_COMPLEX pixelsYX = pixels[2];
                    FLOAT_COMPLEX pixelsYY = pixels[3];
                    pixels[0]  = (pixelsXX * aXX1);
                    pixels[0] += (pixelsXY * aYX1);
                    pixels[1]  = (pixelsXX * aXY1);
                    pixels[1] += (pixelsXY * aYY1);
                    pixels[2]  = (pixelsYX * aXX1);
                    pixels[2] += (pixelsYY * aYX1);
                    pixels[3]  = (pixelsYX * aXY1);
                    pixels[3] += (pixelsYY * aYY1);

                    // Apply aterm to subgrid: A2^H*P
                    // [ aXX2, aYX1;      [ pixels[0], pixels[1];
                    //   aXY1, aYY2 ]  *    pixels[2], pixels[3] ]
                    pixelsXX = pixels[0];
                    pixelsXY = pixels[1];
                    pixelsYX = pixels[2];
                    pixelsYY = pixels[3];
                    pixels[0]  = (pixelsXX * aXX2);
                    pixels[0] += (pixelsYX * aYX2);
                    pixels[1]  = (pixelsXY * aXX2);
                    pixels[1] += (pixelsYY * aYX2);
                    pixels[2]  = (pixelsXX * aXY2);
                    pixels[2] += (pixelsYX * aYY2);
                    pixels[3]  = (pixelsXY * aXY2);
                    pixels[3] += (pixelsYY * aYY2);

                    // Load spheroidal
                    float sph = (*spheroidal)[y][x];

                    // Compute shifted position in subgrid
                    int x_dst = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                    int y_dst = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

                    // Set subgrid value
                    for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        (*subgrid)[s][pol][y_dst][x_dst] = pixels[pol] * sph;
                    }
                } // end x
            } // end y
        } // end s
    } // pragma parallel
} // end kernel_gridder
} // end extern "C"
