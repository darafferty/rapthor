#include <complex>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#include "Types.h"


extern "C" {
void kernel_adder(
    const int jobsize,
    const MetadataType __restrict__ *metadata,
    const SubGridType  __restrict__ *subgrid,
    GridType           __restrict__ *grid
    ) {
    // Iterate all colums of grid
    #pragma omp parallel for schedule(guided)
    for (int row = 0; row < GRIDSIZE; row++) {
        for (int s = 0; s < jobsize; s++) {
            // Load topleft corner of subgrid
            int subgrid_x = metadata[s]->coordinate.x;
            int subgrid_y = metadata[s]->coordinate.y;

            // Compute y offset
            int offset_y = row - subgrid_y;

            // Check wheter subgrid fits in grid and matches curent row
            if (subgrid_x >= 0 && subgrid_x < GRIDSIZE-SUBGRIDSIZE &&
                subgrid_y >= 0 && subgrid_y < GRIDSIZE-SUBGRIDSIZE &&
                 offset_y >= 0 &&  offset_y < SUBGRIDSIZE) {

                // Iterate all columns of subgrid
                for (int x = 0; x < SUBGRIDSIZE; x++) {
                    // Compute shifted position in subgrid
                    int x_src = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                    int y_src = (offset_y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

                    // Add subgrid value to grid
                    (*grid)[0][row][subgrid_x+x] += (*subgrid)[s][0][y_src][x_src];
                    (*grid)[1][row][subgrid_x+x] += (*subgrid)[s][1][y_src][x_src];
                    (*grid)[2][row][subgrid_x+x] += (*subgrid)[s][2][y_src][x_src];
                    (*grid)[3][row][subgrid_x+x] += (*subgrid)[s][3][y_src][x_src];
                }
            }
        }
    }
}
}
