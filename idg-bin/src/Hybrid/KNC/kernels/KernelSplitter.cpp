#pragma omp declare target

#include <complex>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "Types.h"

namespace idg {

void kernel_splitter(
    const int jobsize,
    const void *_metadata,
          void *_subgrid,
    const void *_grid,
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
    for (int s = 0; s < jobsize; s++) {
        // Load position in grid
        int grid_x = metadata[s]->coordinate.x;
        int grid_y = metadata[s]->coordinate.y;

        for (int y = 0; y < subgridsize; y++) {
            for (int x = 0; x < subgridsize; x++) {
                // Compute shifted position in subgrid
                int x_dst = (x + (subgridsize/2)) % subgridsize;
                int y_dst = (y + (subgridsize/2)) % subgridsize;

                // Check wheter subgrid fits in grid
                if (grid_x >= 0 && grid_x < gridsize-subgridsize &&
                    grid_y >= 0 && grid_y < gridsize-subgridsize) {

                    // Set grid value to subgrid
                    for (int pol = 0; pol < nr_polarizations; pol++) {
                        (*subgrid)[s][pol][y_dst][x_dst] = (*grid)[pol][grid_y+y][grid_x+x];
                    }
                }
            }
        }
    }
}
} // end namespace idg

#pragma omp end declare target
