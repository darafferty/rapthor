#include <complex>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "Types.h"


extern "C" {
void kernel_adder(
    const int jobsize,
    const MetadataType __restrict__ *metadata,
    const SubGridType  __restrict__ *subgrid,
    GridType           __restrict__ *grid
    ) {

    #pragma omp parallel for
    for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
        for (int s = 0; s < jobsize; s++) {
            // Load position in grid
            int grid_x = metadata[s]->coordinate.x - (SUBGRIDSIZE/2);
            int grid_y = metadata[s]->coordinate.y - (SUBGRIDSIZE/2);

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

uint64_t kernel_adder_flops(int jobsize) {
    return 1ULL * jobsize * SUBGRIDSIZE * SUBGRIDSIZE * (
    // Shift
    8 +
    // Add
    4
    );
}

uint64_t kernel_adder_bytes(int jobsize) {
    return 1ULL * jobsize * SUBGRIDSIZE * SUBGRIDSIZE * (
    // Coordinate
    2 * sizeof(unsigned) +
    // Pixels
    3 * NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX));
}
}
