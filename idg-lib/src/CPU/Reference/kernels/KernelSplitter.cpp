#include <complex>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "Types.h"


extern "C" {
void kernel_splitter(
    const long           nr_subgrids,
    const long           grid_size,
    const int            subgrid_size,
    const idg::Metadata* metadata,
          idg::float2*   subgrid,
    const idg::float2*   grid
    ) {

    #pragma omp parallel for
    for (int s = 0; s < nr_subgrids; s++) {
        // Load position in grid
        int grid_x = metadata[s].coordinate.x;
        int grid_y = metadata[s].coordinate.y;

        for (int y = 0; y < subgrid_size; y++) {
            for (int x = 0; x < subgrid_size; x++) {
                // Compute shifted position in subgrid
                int x_dst = (x + (subgrid_size/2)) % subgrid_size;
                int y_dst = (y + (subgrid_size/2)) % subgrid_size;

                // Check wheter subgrid fits in grid
                if (grid_x >= 0 && grid_x < grid_size-subgrid_size &&
                    grid_y >= 0 && grid_y < grid_size-subgrid_size) {

                    // Compute phasor
                    float phase  = -M_PI*(x+y-subgrid_size)/subgrid_size;
                    idg::float2 phasor = {cosf(phase), sinf(phase)};

                    // Set grid value to subgrid
                    for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        int src_idx = index_grid(grid_size, pol, grid_y + y, grid_x + x);
                        int dst_idx = index_subgrid(NR_POLARIZATIONS, subgrid_size, s, pol, y_dst, x_dst);
                        subgrid[dst_idx] = phasor * grid[src_idx];
                    }
                }
            }
        }
    }
}
}
