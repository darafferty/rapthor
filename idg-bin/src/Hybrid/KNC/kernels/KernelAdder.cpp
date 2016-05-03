#pragma omp declare target

#include <complex>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "Types.h"

namespace idg {
namespace kernel {
namespace knc {

void adder(
    const int nr_subgrids,
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

    // Iterate all rows of grid
    #pragma omp parallel for schedule(guided)
    for (int row = 0; row < gridsize; row++) {
        for (int s = 0; s < nr_subgrids; s++) {
            // Load topleft corner of subgrid
            int subgrid_x = metadata[s]->coordinate.x;
            int subgrid_y = metadata[s]->coordinate.y;

            // Compute y offset
            int offset_y = row - subgrid_y;

            // Check wheter subgrid fits in grid and matches curent row
            if (subgrid_x >= 0 && subgrid_x < gridsize-subgridsize &&
                subgrid_y >= 0 && subgrid_y < gridsize-subgridsize &&
                 offset_y >= 0 &&  offset_y < subgridsize) {

                // Iterate all columns of subgrid
                for (int x = 0; x < subgridsize; x++) {
                    // Compute shifted position in subgrid
                    int x_src = (x + (subgridsize/2)) % subgridsize;
                    int y_src = (offset_y + (subgridsize/2)) % subgridsize;

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

} // end namespace knc
} // end namespace kernel
} // end namespace idg

#pragma omp end declare target
