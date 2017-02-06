#include <complex>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "Types.h"
#include "Math.h"


extern "C" {
void kernel_splitter(
    const int nr_subgrids,
    const int gridsize,
    const idg::Metadata metadata[],
          idg::float2   subgrid[][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE],
    const idg::float2   grid[]
    ) {
    // Precompute phaosr
    float phasor_real[SUBGRIDSIZE][SUBGRIDSIZE];
    float phasor_imag[SUBGRIDSIZE][SUBGRIDSIZE];
    compute_phasor(phasor_real, phasor_imag);

    #pragma omp parallel for
    for (int s = 0; s < nr_subgrids; s++) {
        // Load position in grid
        int grid_x = metadata[s].coordinate.x;
        int grid_y = metadata[s].coordinate.y;

        for (int y = 0; y < SUBGRIDSIZE; y++) {
            for (int x = 0; x < SUBGRIDSIZE; x++) {
                // Compute shifted position in subgrid
                int x_dst = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                int y_dst = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

                // Check wheter subgrid fits in grid
                if (grid_x >= 0 && grid_x < gridsize-SUBGRIDSIZE &&
                    grid_y >= 0 && grid_y < gridsize-SUBGRIDSIZE) {

                    // Load phasor
                    idg::float2 phasor = {phasor_real[y][x], phasor_imag[y][x]};

                    // Set grid value to subgrid
                    for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
						int grid_idx = (pol * gridsize * gridsize) + ((grid_y + y) * gridsize) + (grid_x + x);
                        subgrid[s][pol][y_dst][x_dst] = phasor * grid[grid_idx];
                    }
                }
            }
        }
    }
}
}
