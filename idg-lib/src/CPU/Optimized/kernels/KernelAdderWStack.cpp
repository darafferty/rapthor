#include <complex>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#include "Types.h"


extern "C" {
void kernel_adder_wstack(
    const int nr_subgrids,
    const int gridsize,
    const int nr_w_layers,
    const idg::Metadata metadata[],
    const idg::float2   subgrid[][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE],
          idg::float2   grid[]
    ) {

    const int transpose[4] = {0, 2, 1, 3};

    // Precompute phaosr
    float phasor_real[SUBGRIDSIZE][SUBGRIDSIZE];
    float phasor_imag[SUBGRIDSIZE][SUBGRIDSIZE];

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < SUBGRIDSIZE; y++) {
        for (int x = 0; x < SUBGRIDSIZE; x++) {
            float phase  = M_PI*(x+y-SUBGRIDSIZE)/SUBGRIDSIZE;
            phasor_real[y][x] = cosf(phase);
            phasor_imag[y][x] = sinf(phase);
        }
    }

    for (int w_layer; w_layer < nr_w_layers; w_layer++){
        idg::float2 *grid_ = grid + w_layer * NR_POLARIZATIONS * gridsize * gridsize;
        // Iterate all colums of grid
        #pragma omp parallel for schedule(guided)
        for (int row = 1; row < gridsize; row++) {
            for (int s = 0; s < nr_subgrids; s++) {
                int subgrid_z = metadata[s].coordinate.z;
                if (subgrid_z < 0)
                {
                    // Load topleft corner of subgrid
                    int subgrid_x = gridsize - metadata[s].coordinate.x - SUBGRIDSIZE + 1;
                    int subgrid_y = gridsize - metadata[s].coordinate.y - SUBGRIDSIZE + 1;
                    int subgrid_w_layer = -subgrid_z-1;

                    if (subgrid_w_layer != w_layer) continue;

                    // Compute y offset
                    int offset_y = row - subgrid_y;
                    int offset_y_mirrored = SUBGRIDSIZE - 1 - offset_y;

                    // Check wheter subgrid fits in grid and matches curent row
                    if (subgrid_x >= 0 && subgrid_x < gridsize-SUBGRIDSIZE &&
                        subgrid_y >= 0 && subgrid_y < gridsize-SUBGRIDSIZE &&
                        offset_y >= 0 &&  offset_y < SUBGRIDSIZE) {

                        // Iterate all columns of subgrid
                        for (int x = 0; x < SUBGRIDSIZE; x++) {
                            int x_mirrored = SUBGRIDSIZE - 1 - x;
                            // Compute shifted position in subgrid
                            int x_src = (x_mirrored + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                            int y_src = (offset_y_mirrored + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

                            // Load phasor
                            idg::float2 phasor = {phasor_real[offset_y_mirrored][x_mirrored], phasor_imag[offset_y_mirrored][x_mirrored]};

                            // Add subgrid value to grid
                            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                                int pol_transposed = transpose[pol];
                                int grid_idx = (pol * gridsize * gridsize) + (row * gridsize) + (subgrid_x + x);
                                grid_[grid_idx] += conj(phasor * subgrid[s][pol_transposed][y_src][x_src]);
                            }
                        }
                    }
                }
                else
                {
                    // Load topleft corner of subgrid
                    int subgrid_x = metadata[s].coordinate.x;
                    int subgrid_y = metadata[s].coordinate.y;
                    int subgrid_w_layer = subgrid_z;

                    if (subgrid_w_layer != w_layer) continue;

                    // Compute y offset
                    int offset_y = row - subgrid_y;

                    // Check wheter subgrid fits in grid and matches curent row
                    if (subgrid_x >= 0 && subgrid_x < gridsize-SUBGRIDSIZE &&
                        subgrid_y >= 0 && subgrid_y < gridsize-SUBGRIDSIZE &&
                        offset_y >= 0 &&  offset_y < SUBGRIDSIZE) {

                        // Iterate all columns of subgrid
                        for (int x = 0; x < SUBGRIDSIZE; x++) {
                            // Compute shifted position in subgrid
                            int x_src = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                            int y_src = (offset_y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

                            // Load phasor
                            idg::float2 phasor = {phasor_real[offset_y][x], phasor_imag[offset_y][x]};

                            // Add subgrid value to grid
                            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                                int grid_idx = (pol * gridsize * gridsize) + (row * gridsize) + (subgrid_x + x);
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
