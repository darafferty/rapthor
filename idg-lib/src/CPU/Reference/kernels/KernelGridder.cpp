#include <iostream>
#include <complex>

#include <math.h>
#include <stdio.h>
#include <immintrin.h>
#include <omp.h>

#include <string.h>
#include <stdint.h>

#include "Types.h"


extern "C" {
    void kernel_gridder(
        const int nr_subgrids,
        const float w_offset,
        const int nr_channels,
        const idg::UVW		uvw[],
        const float         wavenumbers[],
        const idg::float2   visibilities[][NR_POLARIZATIONS],
        const float         spheroidal[SUBGRIDSIZE][SUBGRIDSIZE],
        const idg::float2   aterm[NR_STATIONS][NR_TIMESLOTS][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE],
        const idg::Metadata metadata[],
              idg::float2   subgrid[][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE]
        )
    {
        // Find offset of first subgrid
        const idg::Metadata m = metadata[0];
        const int baseline_offset_1 = m.baseline_offset;
        const int time_offset_1 = m.time_offset; // should be 0

        #pragma omp parallel shared(uvw, wavenumbers, visibilities, spheroidal, aterm, metadata)
        {
            // Iterate all subgrids
            #pragma omp for
            for (int s = 0; s < nr_subgrids; s++) {
                // Load metadata
                const idg::Metadata m = metadata[s];
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

                // Iterate all pixels in subgrid
                for (int y = 0; y < SUBGRIDSIZE; y++) {
                    for (int x = 0; x < SUBGRIDSIZE; x++) {
                        // Initialize pixel for every polarization
                        idg::float2 pixels[NR_POLARIZATIONS];
                        memset(pixels, 0, NR_POLARIZATIONS * sizeof(idg::float2));

                        // Compute l,m,n
                        const float l = (x-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
                        const float m = (y-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
                        // evaluate n = 1.0f - sqrt(1.0 - (l * l) - (m * m));
                        // accurately for small values of l and m
                        const float tmp = (l * l) + (m * m);
                        const float n = tmp / (1.0f + sqrtf(1.0f - tmp));

                        // Iterate all timesteps
                        for (int time = 0; time < nr_timesteps; time++) {
                            // Load UVW coordinates
                            float u = uvw[offset + time].u;
                            float v = uvw[offset + time].v;
                            float w = uvw[offset + time].w;

                            // Compute phase index
                            float phase_index = u*l + v*m + w*n;

                            // Compute phase offset
                            float phase_offset = u_offset*l + v_offset*m + w_offset*n;

                            // Update pixel for every channel
                            for (int chan = 0; chan < nr_channels; chan++) {
                                // Compute phase
                                float wavenumber = wavenumbers[chan];
                                float phase  = (phase_index * wavenumber) - phase_offset;

                                // Compute phasor
                                float phasor_real = cosf(phase);
                                float phasor_imag = sinf(phase);
                                idg::float2 phasor = {phasor_real, phasor_imag};

                                // Update pixel for every polarization
                                size_t index = (offset + time)*nr_channels + chan*NR_POLARIZATIONS;
                                for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                                    idg::float2 visibility = visibilities[index][pol];
                                    pixels[pol] += visibility * phasor;
                                }
                            }
                        }

                        // Load a term for station1
                        idg::float2 aXX1 = aterm[station1][aterm_index][0][y][x];
                        idg::float2 aXY1 = aterm[station1][aterm_index][1][y][x];
                        idg::float2 aYX1 = aterm[station1][aterm_index][2][y][x];
                        idg::float2 aYY1 = aterm[station1][aterm_index][3][y][x];

                        // Load aterm for station2
                        idg::float2 aXX2 = conj(aterm[station2][aterm_index][0][y][x]);
                        idg::float2 aXY2 = conj(aterm[station2][aterm_index][1][y][x]);
                        idg::float2 aYX2 = conj(aterm[station2][aterm_index][2][y][x]);
                        idg::float2 aYY2 = conj(aterm[station2][aterm_index][3][y][x]);

                        // Apply aterm to subgrid: P*A1
                        // [ pixels[0], pixels[1];    [ aXX1, aXY1;
                        //   pixels[2], pixels[3] ] *   aYX1, aYY1 ]
                        idg::float2 pixelsXX = pixels[0];
                        idg::float2 pixelsXY = pixels[1];
                        idg::float2 pixelsYX = pixels[2];
                        idg::float2 pixelsYY = pixels[3];
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
                        float sph = spheroidal[y][x];

                        // Compute shifted position in subgrid
                        int x_dst = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                        int y_dst = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

                        // Set subgrid value
                        for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                            subgrid[s][pol][y_dst][x_dst] = pixels[pol] * sph;
                        }
                    } // end x
                } // end y
            } // end s
        } // end pragma parallel
    }  // end kernel_gridder
}  // end extern C
