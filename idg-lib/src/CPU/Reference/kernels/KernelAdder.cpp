#include <complex>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "Types.h"


extern "C" {
void kernel_adder(
    const long           nr_subgrids,
    const long           grid_size,
    const int            subgrid_size,
    const idg::Metadata* metadata,
    const idg::float2*   subgrid,
          idg::float2*   grid
    ) {

    #pragma omp parallel for
    for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
        for (int s = 0; s < nr_subgrids; s++) {
            // Load position in grid
            int grid_x = metadata[s].coordinate.x;
            int grid_y = metadata[s].coordinate.y;

            // Check wheter subgrid fits in grid
            if (grid_x >= 0 && grid_x < grid_size-subgrid_size &&
                grid_y >= 0 && grid_y < grid_size-subgrid_size) {

                for (int y = 0; y < subgrid_size; y++) {
                    for (int x = 0; x < subgrid_size; x++) {
                        // Compute shifted position in subgrid
                        int x_src = (x + (subgrid_size/2)) % subgrid_size;
                        int y_src = (y + (subgrid_size/2)) % subgrid_size;

                        // Compute phasor
                        float phase  = M_PI*(x+y-subgrid_size)/subgrid_size;
                        idg::float2 phasor = {cosf(phase), sinf(phase)};

                        // Add subgrid value to grid
                        int dst_idx = index_grid(grid_size, pol, grid_y + y, grid_x + x);
                        int src_idx = index_subgrid(NR_POLARIZATIONS, subgrid_size, s, pol, y_src, x_src);
                        grid[dst_idx] += phasor * subgrid[src_idx];
                    }
                }
            }
        }
    }
}
}
