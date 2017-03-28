#include <complex>
#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "Types.h"


extern "C" {
void kernel_splitter_wstack(
    const int nr_subgrids,
    const int gridsize,
    const idg::Metadata metadata[],
          idg::float2   subgrid[][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE],
    const idg::float2   grid[]
    )
{

    const int transpose[4] = {0, 2, 1, 3};

    // Precompute phaosr
    float phasor_real[SUBGRIDSIZE][SUBGRIDSIZE];
    float phasor_imag[SUBGRIDSIZE][SUBGRIDSIZE];

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < SUBGRIDSIZE; y++) {
        for (int x = 0; x < SUBGRIDSIZE; x++) {
            float phase  = -M_PI*(x+y-SUBGRIDSIZE)/SUBGRIDSIZE;
            phasor_real[y][x] = cosf(phase);
            phasor_imag[y][x] = sinf(phase);
        }
    }

    #pragma omp parallel for
    for (int s = 0; s < nr_subgrids; s++) {
        // Load position in grid
        int grid_x = metadata[s].coordinate.x;
        int grid_y = metadata[s].coordinate.y;
        int grid_z = metadata[s].coordinate.z;

        if (grid_z < 0)
        {
            for (int y = 0; y < SUBGRIDSIZE; y++) {
                for (int x = 0; x < SUBGRIDSIZE; x++) {
                    // Compute shifted position in subgrid
                    int x_dst = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                    int y_dst = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

                    // Check wheter subgrid fits in grid
                    if (grid_x >= 1 && grid_x < gridsize-SUBGRIDSIZE &&
                        grid_y >= 1 && grid_y < gridsize-SUBGRIDSIZE) {

                        // Load phasor
                        idg::float2 phasor = {phasor_real[y][x], phasor_imag[y][x]};

                        // Set grid value to subgrid
                        for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                            size_t grid_idx =
                                ( -(grid_z+1) * NR_POLARIZATIONS * size_t(gridsize) * size_t(gridsize)) +
                                (transpose[pol] * size_t(gridsize) * size_t(gridsize)) +
                                ((size_t(gridsize) - grid_y - y) * size_t(gridsize)) +
                                (size_t(gridsize) - grid_x - x);
                            subgrid[s][pol][y_dst][x_dst] = phasor * conj(grid[grid_idx]);
                        }
                    }
                }
            }
        }
        else
        {
            for (int y = 0; y < SUBGRIDSIZE; y++) {
                for (int x = 0; x < SUBGRIDSIZE; x++) {
                    // Compute shifted position in subgrid
                    int x_dst = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                    int y_dst = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

                    // Check wheter subgrid fits in grid
                    if (grid_x >= 1 && grid_x < gridsize-SUBGRIDSIZE &&
                        grid_y >= 1 && grid_y < gridsize-SUBGRIDSIZE) {

                        // Load phasor
                        idg::float2 phasor = {phasor_real[y][x], phasor_imag[y][x]};

                        // Set grid value to subgrid
                        for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                            size_t grid_idx =
                                (grid_z * NR_POLARIZATIONS * size_t(gridsize) * size_t(gridsize)) +
                                (pol * size_t(gridsize) * size_t(gridsize)) +
                                ((grid_y + y) * size_t(gridsize)) +
                                (grid_x + x);
                            subgrid[s][pol][y_dst][x_dst] = phasor * grid[grid_idx];
                        }
                    }
                }
            }
        }
    }
}
}
