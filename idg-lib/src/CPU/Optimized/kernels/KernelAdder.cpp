#include <complex>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#include "Types.h"


extern "C" {
void kernel_adder(
    const long           nr_subgrids,
    const long           grid_size,
    const int            subgrid_size,
    const idg::Metadata* metadata,
    const idg::float2*   subgrid,
          idg::float2*   grid)
{
    // Precompute phaosr
    float phasor_real[subgrid_size][subgrid_size];
    float phasor_imag[subgrid_size][subgrid_size];

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < subgrid_size; y++) {
        for (int x = 0; x < subgrid_size; x++) {
            float phase  = M_PI*(x+y-subgrid_size)/subgrid_size;
            phasor_real[y][x] = cosf(phase);
            phasor_imag[y][x] = sinf(phase);
        }
    }

    // Iterate all rows of grid
    #pragma omp parallel for schedule(guided)
    for (int row = 0; row < grid_size; row++) {
        for (int s = 0; s < nr_subgrids; s++) {
            // Load topleft corner of subgrid
            int subgrid_x = metadata[s].coordinate.x;
            int subgrid_y = metadata[s].coordinate.y;

            // Compute y offset
            int offset_y = row - subgrid_y;

            // Check wheter subgrid fits in grid and matches curent row
            if (subgrid_x >= 0 && subgrid_x < grid_size-subgrid_size &&
                subgrid_y >= 0 && subgrid_y < grid_size-subgrid_size &&
                 offset_y >= 0 &&  offset_y < subgrid_size) {

                // Iterate all columns of subgrid
                for (int x = 0; x < subgrid_size; x++) {
                    // Compute shifted position in subgrid
                    int x_src = (x + (subgrid_size/2)) % subgrid_size;
                    int y_src = (offset_y + (subgrid_size/2)) % subgrid_size;

                    // Load phasor
                    idg::float2 phasor = {phasor_real[offset_y][x], phasor_imag[offset_y][x]};

                    // Add subgrid value to grid
					for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        long dst_idx = index_grid(grid_size, pol, row, subgrid_x + x);
                        long src_idx = index_subgrid(NR_POLARIZATIONS, subgrid_size, s, pol, y_src, x_src);
                        grid[dst_idx] += phasor * subgrid[src_idx];
					} // end for pol
                } // end for row
            } // end if fit
        } // end for s
    } // end for row
} // end kernel_adder
} // end extern "C"
