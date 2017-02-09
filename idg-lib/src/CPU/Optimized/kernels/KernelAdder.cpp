#include <complex>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#include "Types.h"


extern "C" {
void kernel_adder(
    const int nr_subgrids,
    const int gridsize,
    const idg::Metadata metadata[],
    const idg::float2   subgrid[][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE],
          idg::float2   grid[]
    ) {
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

    // Iterate all colums of grid
    #pragma omp parallel for schedule(guided)
    for (int row = 0; row < gridsize; row++) {
        for (int s = 0; s < nr_subgrids; s++) {
            // Load topleft corner of subgrid
            int subgrid_x = metadata[s].coordinate.x;
            int subgrid_y = metadata[s].coordinate.y;

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
						grid[grid_idx] += phasor * subgrid[s][pol][y_src][x_src];
					}
                }
            }
        }
    }
}
}
