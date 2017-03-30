#include <complex>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#include "Types.h"

#include <iostream>


extern "C" {
void kernel_adder_wstack(
    const long nr_subgrids,
    const long grid_size,
    const int subgrid_size,
    const int nr_w_layers,
    const idg::Metadata metadata[],
    const idg::float2   subgrid[][NR_POLARIZATIONS][subgrid_size][subgrid_size],
          idg::float2   grid[]
    ) 
{
    const int transpose[4] = {0, 2, 1, 3};

    // Precompute phaosr
    float phasor_real[subgrid_size][subgrid_size];
    float phasor_imag[subgrid_size][subgrid_size];

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < subgrid_size; y++) {
        for (int x = 0; x < subgrid_size; x++) {
            float phase  = M_PI*(x+y-subgrid_size)/subgrid_size;
            phasor_real[y][x] = cosf(phase);
            phasor_imag[y][x] = sinf(phase);
        }
    }

    for (int w_layer = 0; w_layer < nr_w_layers; w_layer++) {
        idg::float2 *grid_ = grid + w_layer * NR_POLARIZATIONS * size_t(grid_size) * size_t(grid_size);
        // Iterate all colums of grid
        #pragma omp parallel for schedule(guided)
        for (int row = 1; row < grid_size; row++) {
            for (int s = 0; s < nr_subgrids; s++) {
                int subgrid_z = metadata[s].coordinate.z;
                if (subgrid_z < 0) {
                    // Load topleft corner of subgrid
                    int subgrid_x = grid_size - metadata[s].coordinate.x - subgrid_size + 1;
                    int subgrid_y = grid_size - metadata[s].coordinate.y - subgrid_size + 1;
                    int subgrid_w_layer = -subgrid_z-1;

                    if (subgrid_w_layer != w_layer) continue;

                    // Compute y offset
                    int offset_y = row - subgrid_y;
                    int offset_y_mirrored = subgrid_size - 1 - offset_y;

                    // Check wheter subgrid fits in grid and matches curent row
                    if (subgrid_x >= 1 && subgrid_x < grid_size-subgrid_size &&
                        subgrid_y >= 1 && subgrid_y < grid_size-subgrid_size &&
                        offset_y >= 0 &&  offset_y < subgrid_size) {

                        // Iterate all columns of subgrid
                        for (int x = 0; x < subgrid_size; x++) {
                            int x_mirrored = subgrid_size - 1 - x;
                            // Compute shifted position in subgrid
                            int x_src = (x_mirrored + (subgrid_size/2)) % subgrid_size;
                            int y_src = (offset_y_mirrored + (subgrid_size/2)) % subgrid_size;

                            // Load phasor
                            idg::float2 phasor = {phasor_real[offset_y_mirrored][x_mirrored],
                                                  phasor_imag[offset_y_mirrored][x_mirrored]};

                            // Add subgrid value to grid
                            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                                int pol_transposed = transpose[pol];
                                size_t grid_idx = (pol * grid_size * grid_size) + (row * grid_size) + (subgrid_x + x);
                                grid_[grid_idx] += conj(phasor * subgrid[s][pol_transposed][y_src][x_src]);
                            }
                        }
                    }
                } else {
                    // Load topleft corner of subgrid
                    int subgrid_x = metadata[s].coordinate.x;
                    int subgrid_y = metadata[s].coordinate.y;
                    int subgrid_w_layer = subgrid_z;

                    if (subgrid_w_layer != w_layer) continue;

                    // Compute y offset
                    int offset_y = row - subgrid_y;

                    // Check wheter subgrid fits in grid and matches curent row
                    if (subgrid_x >= 1 && subgrid_x < grid_size-subgrid_size &&
                        subgrid_y >= 1 && subgrid_y < grid_size-subgrid_size &&
                        offset_y >= 0 &&  offset_y < subgrid_size) {

                        // Iterate all columns of subgrid
                        for (int x = 0; x < subgrid_size; x++) {
                            // Compute shifted position in subgrid
                            int x_src = (x + (subgrid_size/2)) % subgrid_size;
                            int y_src = (offset_y + (subgrid_size/2)) % subgrid_size;

                            // Load phasor
                            idg::float2 phasor = {phasor_real[offset_y][x], phasor_imag[offset_y][x]};

                            // Add subgrid value to grid
                            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                                size_t grid_idx = (pol * size_t(grid_size) * size_t(grid_size)) + (row * size_t(grid_size)) + (subgrid_x + x);
                                grid_[grid_idx] += phasor * subgrid[s][pol][y_src][x_src];
                            }
                        }
                    }
                }
            }
        }
    }
}
}
