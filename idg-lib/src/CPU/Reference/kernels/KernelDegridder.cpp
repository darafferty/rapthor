#include <complex>

#include <math.h>
#include <stdio.h>
#include <immintrin.h>
#include <string.h>
#include <stdint.h>

#include "Types.h"


extern "C" {
    void kernel_degridder(
        const int nr_subgrids,
        const float w_offset,
        const int nr_channels,
        const idg::UVW		uvw[],
        const float         wavenumbers[],
              idg::float2   visibilities[][NR_POLARIZATIONS],
        const float         spheroidal[SUBGRIDSIZE][SUBGRIDSIZE],
        const idg::float2   aterm[NR_STATIONS][NR_TIMESLOTS][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE],
        const idg::Metadata metadata[],
        const idg::float2   subgrid[][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE]
        )
    {
        // Find offset of first subgrid
        const idg::Metadata m = metadata[0];
        const int baseline_offset_1 = m.baseline_offset;
        const int time_offset_1 = m.time_offset;

        #pragma omp parallel shared(uvw, wavenumbers, visibilities, spheroidal, aterm, metadata)
        {
            // Iterate all subgrids
            #pragma omp for
            for (int s = 0; s < nr_subgrids; s++) {

                // Load metadata
                const idg::Metadata m = metadata[s];
                const int local_offset = (m.baseline_offset - baseline_offset_1) +
                    (m.time_offset - time_offset_1);
                const int nr_timesteps = m.nr_timesteps;
                const int aterm_index = m.aterm_index;
                const int station1 = m.baseline.station1;
                const int station2 = m.baseline.station2;
                const int x_coordinate = m.coordinate.x;
                const int y_coordinate = m.coordinate.y;

                // Storage
                idg::float2 pixels[SUBGRIDSIZE][SUBGRIDSIZE][NR_POLARIZATIONS];

                // Apply aterm to subgrid
                for (int y = 0; y < SUBGRIDSIZE; y++) {
                    for (int x = 0; x < SUBGRIDSIZE; x++) {
                        // Load aterm for station1
                        idg::float2 aXX1 = aterm[station1][aterm_index][0][y][x];
                        idg::float2 aXY1 = aterm[station1][aterm_index][1][y][x];
                        idg::float2 aYX1 = aterm[station1][aterm_index][2][y][x];
                        idg::float2 aYY1 = aterm[station1][aterm_index][3][y][x];

                        // Load aterm for station2
                        idg::float2 aXX2 = conj(aterm[station2][aterm_index][0][y][x]);
                        idg::float2 aXY2 = conj(aterm[station2][aterm_index][1][y][x]);
                        idg::float2 aYX2 = conj(aterm[station2][aterm_index][2][y][x]);
                        idg::float2 aYY2 = conj(aterm[station2][aterm_index][3][y][x]);

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

                        // Apply aterm to subgrid
                        pixels[y][x][0]  = pixelsXX * aXX1;
                        pixels[y][x][0] += pixelsXY * aYX1;
                        pixels[y][x][1]  = pixelsXX * aXY1;
                        pixels[y][x][1] += pixelsXY * aYY1;
                        pixels[y][x][2]  = pixelsYX * aXX1;
                        pixels[y][x][2] += pixelsYY * aYX1;
                        pixels[y][x][3]  = pixelsYX * aXY1;
                        pixels[y][x][3] += pixelsYY * aYY1;

                        pixelsXX = pixels[y][x][0];
                        pixelsXY = pixels[y][x][1];
                        pixelsYX = pixels[y][x][2];
                        pixelsYY = pixels[y][x][3];
                        pixels[y][x][0]  = pixelsXX * aXX2;
                        pixels[y][x][0] += pixelsYX * aYX2;
                        pixels[y][x][1]  = pixelsXY * aXX2;
                        pixels[y][x][1] += pixelsYY * aYX2;
                        pixels[y][x][2]  = pixelsXX * aXY2;
                        pixels[y][x][2] += pixelsYX * aYY2;
                        pixels[y][x][3]  = pixelsXY * aXY2;
                        pixels[y][x][3] += pixelsYY * aYY2;
                    } // end x
                } // end y

                // Compute u and v offset in wavelenghts
                const float u_offset = (x_coordinate + SUBGRIDSIZE/2 - GRIDSIZE/2)
                                       * (2*M_PI / IMAGESIZE);
                const float v_offset = (y_coordinate + SUBGRIDSIZE/2 - GRIDSIZE/2)
                                       * (2*M_PI / IMAGESIZE);

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
                                float wavenumber = wavenumbers[chan];
                                float phase  = phase_offset - (phase_index * wavenumber);

                                // Compute phasor
                                float phasor_real = cosf(phase);
                                float phasor_imag = sinf(phase);

                                idg::float2 phasor = {phasor_real, phasor_imag};

                                for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                                    sum[pol] += pixels[y][x][pol] * phasor;
                                }
                            }
                        }

                        const float scale = 1.0f / (SUBGRIDSIZE*SUBGRIDSIZE);
                        size_t index = (local_offset + time)*nr_channels + chan*NR_POLARIZATIONS;
                        for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                            visibilities[index][pol] = sum[pol] * scale;
                        }

                    } // end for channel
                } // end for time
            } // end for s
        } // end #pragma parallel
    } // end kernel_degridder
} // end extern "C"
