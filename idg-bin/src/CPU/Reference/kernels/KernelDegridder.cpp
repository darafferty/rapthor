#include <complex>

#include <math.h>
#include <stdio.h>
#include <immintrin.h>
#include <string.h>
#include <stdint.h>

#include "Types.h"


extern "C" {
    void kernel_degridder(
        const int           nr_subgrids,
        const float         w_offset_in_lambda,
        const int           nr_channels,
        const idg::UVW		uvw[],
        const float         wavenumbers[],
              idg::float2   visibilities[][NR_POLARIZATIONS],
        const float         spheroidal[SUBGRIDSIZE][SUBGRIDSIZE],
        const idg::float2   aterm[][NR_STATIONS][SUBGRIDSIZE][SUBGRIDSIZE][NR_POLARIZATIONS],
        const idg::Metadata metadata[],
        const idg::float2   subgrid[][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE]
        )
    {
        // Find offset of first subgrid
        const idg::Metadata m       = metadata[0];
        const int baseline_offset_1 = m.baseline_offset;
        const int time_offset_1     = m.time_offset;

        #pragma omp parallel shared(uvw, wavenumbers, visibilities, spheroidal, aterm, metadata)
        {
            // Iterate all subgrids
            #pragma omp for
            for (int s = 0; s < nr_subgrids; s++) {

                // Load metadata
                const idg::Metadata m  = metadata[s];
                const int local_offset = (m.baseline_offset - baseline_offset_1) +
                                         (m.time_offset - time_offset_1);
                const int nr_timesteps = m.nr_timesteps;
                const int aterm_index  = m.aterm_index;
                const int station1     = m.baseline.station1;
                const int station2     = m.baseline.station2;
                const int x_coordinate = m.coordinate.x;
                const int y_coordinate = m.coordinate.y;

                // Storage
                idg::float2 pixels[SUBGRIDSIZE][SUBGRIDSIZE][NR_POLARIZATIONS];

                // Apply aterm to subgrid
                for (int y = 0; y < SUBGRIDSIZE; y++) {
                    for (int x = 0; x < SUBGRIDSIZE; x++) {
                        // Load aterm for station1
                        idg::float2 aXX1 = conj(aterm[aterm_index][station1][y][x][0]);
                        idg::float2 aXY1 = conj(aterm[aterm_index][station1][y][x][1]);
                        idg::float2 aYX1 = conj(aterm[aterm_index][station1][y][x][2]);
                        idg::float2 aYY1 = conj(aterm[aterm_index][station1][y][x][3]);

                        // Load aterm for station2
                        idg::float2 aXX2 = aterm[aterm_index][station2][y][x][0];
                        idg::float2 aXY2 = aterm[aterm_index][station2][y][x][1];
                        idg::float2 aYX2 = aterm[aterm_index][station2][y][x][2];
                        idg::float2 aYY2 = aterm[aterm_index][station2][y][x][3];

                        // Load spheroidal
                        float _spheroidal = spheroidal[y][x];

                        // Compute shifted position in subgrid
                        int x_src = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                        int y_src = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

                        // Load uv values
                        idg::float2 pixelsXX = _spheroidal * subgrid[s][0][y_src][x_src];
                        idg::float2 pixelsXY = _spheroidal * subgrid[s][1][y_src][x_src];
                        idg::float2 pixelsYX = _spheroidal * subgrid[s][2][y_src][x_src];
                        idg::float2 pixelsYY = _spheroidal * subgrid[s][3][y_src][x_src];

                        // Apply aterm to subgrid: P*A1^H
                        // [ pixels[0], pixels[1];    [ conj(aXX1), conj(aYX1);
                        //   pixels[2], pixels[3] ] *   conj(aXY1), conj(aYY1) ]
                        pixels[y][x][0]  = pixelsXX * aXX1;
                        pixels[y][x][0] += pixelsXY * aXY1;
                        pixels[y][x][1]  = pixelsXX * aYX1;
                        pixels[y][x][1] += pixelsXY * aYY1;
                        pixels[y][x][2]  = pixelsYX * aXX1;
                        pixels[y][x][2] += pixelsYY * aXY1;
                        pixels[y][x][3]  = pixelsYX * aYX1;
                        pixels[y][x][3] += pixelsYY * aYY1;

                        // Apply aterm to subgrid: A2*P
                        // [ aXX2, aXY1;      [ pixels[0], pixels[1];
                        //   aYX1, aYY2 ]  *    pixels[2], pixels[3] ]
                        pixelsXX = pixels[y][x][0];
                        pixelsXY = pixels[y][x][1];
                        pixelsYX = pixels[y][x][2];
                        pixelsYY = pixels[y][x][3];
                        pixels[y][x][0]  = pixelsXX * aXX2;
                        pixels[y][x][0] += pixelsYX * aXY2;
                        pixels[y][x][1]  = pixelsXY * aXX2;
                        pixels[y][x][1] += pixelsYY * aXY2;
                        pixels[y][x][2]  = pixelsXX * aYX2;
                        pixels[y][x][2] += pixelsYX * aYY2;
                        pixels[y][x][3]  = pixelsXY * aYX2;
                        pixels[y][x][3] += pixelsYY * aYY2;
                    } // end x
                } // end y

                // Compute u and v offset in wavelenghts
                const float u_offset = (x_coordinate + SUBGRIDSIZE/2 - GRIDSIZE/2)
                                       * (2*M_PI / IMAGESIZE);
                const float v_offset = (y_coordinate + SUBGRIDSIZE/2 - GRIDSIZE/2)
                                       * (2*M_PI / IMAGESIZE);
                const float w_offset = 2*M_PI * w_offset_in_lambda; // TODO: check!

                // Iterate all timesteps
                for (int time = 0; time < nr_timesteps; time++) {
                    // Load UVW coordinates
                    float u = uvw[local_offset + time].u;
                    float v = uvw[local_offset + time].v;
                    float w = uvw[local_offset + time].w;

                    // Iterate all channels
                    for (int chan = 0; chan < nr_channels; chan++) {

                        // Update all polarizations
                        idg::float2 sum[NR_POLARIZATIONS];
                        memset(sum, 0, NR_POLARIZATIONS * sizeof(idg::float2));

                        // Iterate all pixels in subgrid
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
                                float phase_index = u*l + v*m + w*n;

                                // Compute phase offset
                                float phase_offset = u_offset*l + v_offset*m + w_offset*n;

                                // Compute phase
                                float phase = (phase_index * wavenumbers[chan]) - phase_offset;

                                // Compute phasor
                                idg::float2 phasor = {cosf(phase), sinf(phase)};

                                for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                                    sum[pol] += pixels[y][x][pol] * phasor;
                                }
                            }
                        }

                        const float scale = 1.0f / (SUBGRIDSIZE*SUBGRIDSIZE);
                        size_t index = (local_offset + time)*nr_channels + chan;
                        for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                            visibilities[index][pol] = sum[pol] * scale;
                        }

                    } // end for channel
                } // end for time
            } // end for s
        } // end #pragma parallel
    } // end kernel_degridder
} // end extern "C"
