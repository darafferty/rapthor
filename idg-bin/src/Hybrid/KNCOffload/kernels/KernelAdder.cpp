#pragma omp declare target

#include <complex>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "Types.h"

namespace idg {

void kernel_adder(
    const int jobsize,
    const void *_metadata,
    const void *_subgrid,
          void *_grid,
    const int gridsize,
    const int subgridsize,
    const int nr_polarizations)
    {
    TYPEDEF_BASELINE
    TYPEDEF_COORDINATE
    TYPEDEF_METADATA
    TYPEDEF_METADATA_TYPE
    TYPEDEF_SUBGRID_TYPE
    TYPEDEF_GRID_TYPE
    
    MetadataType *metadata = (MetadataType *) _metadata;
    SubGridType *subgrid = (SubGridType *) _subgrid;
    GridType *grid = (GridType *) _grid;

    #pragma omp parallel for
    for (int pol = 0; pol < nr_polarizations; pol++) {
        for (int s = 0; s < jobsize; s++) {
            // Load position in grid
            int grid_x = metadata[s]->coordinate.x;
            int grid_y = metadata[s]->coordinate.y;

            // Check wheter subgrid fits in grid
            if (grid_x >= 0 && grid_x < gridsize-subgridsize &&
                grid_y >= 0 && grid_y < gridsize-subgridsize) {

                for (int y = 0; y < subgridsize; y++) {
                    for (int x = 0; x < subgridsize; x++) {
                        // Compute shifted position in subgrid
                        int x_src = (x + (subgridsize/2)) % subgridsize;
                        int y_src = (y + (subgridsize/2)) % subgridsize;

                        // Add subgrid value to grid
                        (*grid)[pol][grid_y+y][grid_x+x] += (*subgrid)[s][pol][y_src][x_src];
                    }
                }
            }
        }
    }
}

    uint64_t kernel_adder_flops(int jobsize, int subgridsize) {
    return 1ULL * jobsize * subgridsize * subgridsize * (
    // Shift
    8 +
    // Add
    4
    );
}

    uint64_t kernel_adder_bytes(int jobsize, int subgridsize, int nr_polarizations) {
    return 1ULL * jobsize * subgridsize * subgridsize * (
    // Coordinate
    2 * sizeof(unsigned) +
    // Pixels
    3 * nr_polarizations * sizeof(FLOAT_COMPLEX));
}

} // end namespace idg

#pragma omp end declare target
