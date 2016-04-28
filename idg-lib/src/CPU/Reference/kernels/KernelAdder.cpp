#include <complex>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "Types.h"


extern "C" {
void kernel_adder(
    const int nr_subgrids,
    const MetadataType __restrict__ *metadata,
    const SubGridType  __restrict__ *subgrid,
    GridType           __restrict__ *grid
    ) {

    #pragma omp parallel for
    for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
        for (int s = 0; s < nr_subgrids; s++) {
            // Load position in grid
            int grid_x = metadata[s]->coordinate.x;
            int grid_y = metadata[s]->coordinate.y;

            // Check wheter subgrid fits in grid
            if (grid_x >= 0 && grid_x < GRIDSIZE-SUBGRIDSIZE &&
                grid_y >= 0 && grid_y < GRIDSIZE-SUBGRIDSIZE) {

                for (int y = 0; y < SUBGRIDSIZE; y++) {
                    for (int x = 0; x < SUBGRIDSIZE; x++) {
                        // Compute shifted position in subgrid
                        int x_src = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                        int y_src = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

                        // Add subgrid value to grid
                        (*grid)[pol][grid_y+y][grid_x+x] += (*subgrid)[s][pol][y_src][x_src];
                    }
                }
            }
        }
    }
}
}
