#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "Types.h"


extern "C" {
void kernel_adder(
    const int jobsize,
    const UVWType     __restrict__ *uvw,
    const SubGridType __restrict__ *subgrid,
    GridType          __restrict__ *grid
    ) {
    #pragma omp parallel
    {
    #if USE_LIKWID
    likwid_markerThreadInit();
    likwid_markerStartRegion("adder");
    #endif
    #pragma omp for
    for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
        for (int bl = 0; bl < jobsize; bl++) {
            for (int chunk = 0; chunk < CHUNKSIZE; chunk++) {
                // Get first and last UVW coordinate
                int time_offset = chunk * CHUNKSIZE;
		        UVW uvw_first = (*uvw)[bl][time_offset];
		        UVW uvw_last  = (*uvw)[bl][time_offset + CHUNKSIZE];
		
		        // Compute position in master grid
		        int grid_x = ((uvw_first.u + uvw_last.u) / 2) - (SUBGRIDSIZE / 2);
		        int grid_y = ((uvw_first.v + uvw_last.v) / 2) - (SUBGRIDSIZE / 2);
		        
                for (int y = 0; y < SUBGRIDSIZE; y++) {
                    #pragma ivdep
                    for (int x = 0; x < SUBGRIDSIZE; x++) {
                        // Compute shifted position in subgrid
                        int x_src = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                        int y_src = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

                        #if ORDER == ORDER_BL_V_U_P
                        (*grid)[pol][grid_y+y][grid_x+x] += (*subgrid)[bl][chunk][y_src][x_src][pol];
                        #elif ORDER == ORDER_BL_P_V_U
                        (*grid)[pol][grid_y+y][grid_x+x] += (*subgrid)[bl][chunk][pol][y_src][x_src];
                        #endif
                    }
                }
            }
        }
    }
    #if USE_LIKWID
    likwid_markerStopRegion("adder");
    #endif
    }
}

uint64_t kernel_adder_flops(int jobsize) {
    return 1ULL * NR_CHUNKS * SUBGRIDSIZE * SUBGRIDSIZE * jobsize * (
    // Shift
    11 +
    // Add
    4
    );
}

uint64_t kernel_adder_bytes(int jobsize) {
	return 1ULL * NR_CHUNKS * SUBGRIDSIZE * SUBGRIDSIZE * jobsize * (
    // Coordinate
    2 * sizeof(unsigned) +
    // Pixels
    3 * NR_POLARIZATIONS * sizeof(float complex));
}
}
