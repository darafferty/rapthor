#include <complex>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "Types.h"


extern "C" {
void kernel_splitter(
    const int jobsize,
    const UVWType  __restrict__ *uvw,
    SubGridType    __restrict__ *subgrid,
    const GridType __restrict__ *grid
    ) {

    #pragma omp parallel
    {
    #pragma omp for
    for (int bl = 0; bl < jobsize; bl++) {
        for (int chunk = 0; chunk < NR_CHUNKS; chunk++) {
            // Get first and last UVW coordinate
            int time_offset = chunk * CHUNKSIZE;
		    UVW uvw_first = (*uvw)[bl][time_offset];
		    UVW uvw_last  = (*uvw)[bl][time_offset + CHUNKSIZE];
		
		    // Compute position in master grid
		    int grid_x = ((uvw_first.u + uvw_last.u) / 2) - (SUBGRIDSIZE / 2);
		    int grid_y = ((uvw_first.v + uvw_last.v) / 2) - (SUBGRIDSIZE / 2);
        
            for (int y = 0; y < SUBGRIDSIZE; y++) {
                for (int x = 0; x < SUBGRIDSIZE; x++) {
                    // Compute shifted position in subgrid
                    int x_dst = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                    int y_dst = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

                    // Set grid value to subgrid
                    for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        (*subgrid)[bl][chunk][pol][y_dst][x_dst] = (*grid)[pol][grid_y+y][grid_x+x];
                    }
                }
            }
        }
    }
    }
}

uint64_t kernel_splitter_flops(int jobsize) {
    return 1ULL * NR_CHUNKS * SUBGRIDSIZE * SUBGRIDSIZE * jobsize * (
    // Shift
    11 +
    // Add
    4
    );
}

uint64_t kernel_splitter_bytes(int jobsize) {
	return 1ULL * NR_CHUNKS * SUBGRIDSIZE * SUBGRIDSIZE * jobsize * (
    // Coordinate
    2 * sizeof(unsigned) +
    // Pixels
    3 * NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX));
}
}
