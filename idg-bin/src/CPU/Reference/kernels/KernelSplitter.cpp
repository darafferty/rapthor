#include <complex>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "Types.h"


extern "C" {
void kernel_splitter(
    const int jobsize,
    const MetadataType __restrict__ *metadata,
    SubGridType        __restrict__ *subgrid,
    const GridType     __restrict__ *grid
    ) {

    #pragma omp parallel for
    for (int s = 0; s < jobsize; s++) {
        // Load position in grid
        int grid_x = metadata[s]->coordinate.x;
        int grid_y = metadata[s]->coordinate.y;

        for (int y = 0; y < SUBGRIDSIZE; y++) {
            for (int x = 0; x < SUBGRIDSIZE; x++) {
                // Compute shifted position in subgrid
                int x_dst = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                int y_dst = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

                // Check wheter subgrid fits in grid
                if (grid_x >= 0 && grid_x < GRIDSIZE-SUBGRIDSIZE &&
                    grid_y >= 0 && grid_y < GRIDSIZE-SUBGRIDSIZE) {

                    // Set grid value to subgrid
                    for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        (*subgrid)[s][pol][y_dst][x_dst] = (*grid)[pol][grid_y+y][grid_x+x];
                    }
                }
            }
        }
    }
}
}
