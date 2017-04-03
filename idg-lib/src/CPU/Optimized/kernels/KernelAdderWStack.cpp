#include <complex>

#include <stdlib.h>
#include <stdint.h>

#include "Types.h"


extern "C" {
void kernel_adder_wstack(
    const long           nr_subgrids,
    const long           grid_size,
    const int            subgrid_size,
    const int            nr_w_layers,
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

    // Iterate all w-layers
    for (int w_layer = 0; w_layer < nr_w_layers; w_layer++) {
        // Iterate all rows of grid
        #pragma omp parallel for schedule(guided)
        for (int row = 1; row < grid_size; row++) {
            for (int s = 0; s < nr_subgrids; s++) {
                // Load subgrid coordinates
                int subgrid_x = grid_size - metadata[s].coordinate.x - subgrid_size + 1;
                int subgrid_y = grid_size - metadata[s].coordinate.y - subgrid_size + 1;
                int subgrid_w = metadata[s].coordinate.z;

                // Mirror subgrid coordinates for negative w-values
                bool negative_w = subgrid_w < 0;
                if (negative_w) {
                    subgrid_x = grid_size - subgrid_x - subgrid_size + 1;
                    subgrid_y = grid_size - subgrid_y - subgrid_size + 1;
                    subgrid_w =  -subgrid_w - 1;
                }

                // Check whether subgrid matches current w-layer
                if (subgrid_w != w_layer) { continue; };

                // Compute row index of subgrid
                int y          = row - subgrid_y;
                int y_mirrored = subgrid_size - 1 - y;

                // Determine polarization index
                const int index_pol_default[NR_POLARIZATIONS]    = {0, 1, 2, 3};
                const int index_pol_transposed[NR_POLARIZATIONS] = {0, 2, 1, 3};
                int *index_pol = (int *) (negative_w ? index_pol_default : index_pol_transposed);

                // Check whether subgrid fits in grid and matches curent row
                if (subgrid_x >= 1 && subgrid_x < grid_size-subgrid_size &&
                    subgrid_y >= 1 && subgrid_y < grid_size-subgrid_size &&
                    y >= 0 && y < subgrid_size) {

                    // Iterate all columns of subgrid
                    for (int x = 0; x < subgrid_size; x++) {
                        int x_mirrored = subgrid_size - 1 - x;

                        // Compute position in subgrid
                        int x_ = negative_w ? x_mirrored : x;
                        int y_ = negative_w ? y_mirrored : y;
                        int x_src = (x_ + (subgrid_size/2)) % subgrid_size;
                        int y_src = (y_ + (subgrid_size/2)) % subgrid_size;

                        // Compute position in grid
                        int x_dst = subgrid_x + x;
                        int y_dst = row;

                        // Load phasor
                        idg::float2 phasor = {phasor_real[y_][x_], phasor_imag[y_][x_]};

                        // Add subgrid value to grid
                        for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                            int pol_idx = index_pol[pol];
                            int dst_idx = index_grid(
                                NR_POLARIZATIONS, grid_size, w_layer, pol_idx, y_dst, x_dst);
                            int src_idx = index_subgrid(
                                NR_POLARIZATIONS, subgrid_size, s, pol_idx, y_src, x_src);
                            idg::float2 value = phasor * subgrid[src_idx];
                            value = negative_w ? conj(value) : value;
                            grid[dst_idx] += value;
                        } // end for pol
                    } // end for x
                } // end if fit
            } // end for s
        } // end for row
    } // end for w-layer
} // end kernel_adder_wstack
} // end extern "C"
