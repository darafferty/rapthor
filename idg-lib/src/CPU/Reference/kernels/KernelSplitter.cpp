#include <complex>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "Types.h"


extern "C" {
void kernel_splitter(
    const long nr_subgrids,
    const long gridsize,
    const idg::Metadata metadata[],
          idg::float2   subgrid[][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE],
    const idg::float2   grid[]
    ) {

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

                    // Compute phasor
                    float phase  = -M_PI*(x+y-SUBGRIDSIZE)/SUBGRIDSIZE;
                    idg::float2 phasor = {cosf(phase), sinf(phase)};

                    // Set grid value to subgrid
                    for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
						size_t grid_idx = (pol * gridsize * gridsize) + ((grid_y + y) * gridsize) + (grid_x + x);
                        subgrid[s][pol][y_dst][x_dst] = phasor * grid[grid_idx];
                    }
                }
            }
        }
    }
}
}
