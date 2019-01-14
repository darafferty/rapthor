#include <complex>

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
    const idg::float2*   grid)
{
    // Precompute phaosr
    float phasor_real[subgrid_size][subgrid_size];
    float phasor_imag[subgrid_size][subgrid_size];

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < subgrid_size; y++) {
        for (int x = 0; x < subgrid_size; x++) {
            float phase  = -M_PI*(x+y-subgrid_size)/subgrid_size;
            phasor_real[y][x] = cosf(phase);
            phasor_imag[y][x] = sinf(phase);
        }
    }

    #pragma omp parallel for
    for (int s = 0; s < nr_subgrids; s++) {

        // Load subgrid coordinates
        int subgrid_x = metadata[s].coordinate.x;
        int subgrid_y = metadata[s].coordinate.y;
        int subgrid_w = metadata[s].coordinate.z;

        for (int y = 0; y < subgrid_size; y++) {
            for (int x = 0; x < subgrid_size; x++) {

                // Compute position in subgrid
                int x_dst = (x + (subgrid_size/2)) % subgrid_size;
                int y_dst = (y + (subgrid_size/2)) % subgrid_size;

                // Compute position in grid
                int x_src = subgrid_x + x;
                int y_src = subgrid_y + y;

                // Check whether subgrid fits in grid
                if (subgrid_x >= 1 && subgrid_x < grid_size-subgrid_size &&
                    subgrid_y >= 1 && subgrid_y < grid_size-subgrid_size) {

                    // Load phasor
                    idg::float2 phasor = {phasor_real[y][x], phasor_imag[y][x]};

                    // Set grid value to subgrid
                    for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        long src_idx = index_grid(NR_POLARIZATIONS, grid_size, subgrid_w, pol, y_src, x_src);
                        long dst_idx = index_subgrid(NR_POLARIZATIONS, subgrid_size, s, pol, y_dst, x_dst);
                        subgrid[dst_idx] = phasor * grid[src_idx];
                    } // end for pol
                } // end if fit
            } // end for x
        } // end for y
    } // end for s
} // end kernel_splitter
} // end extern "C"
