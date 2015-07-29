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

  //  printf("Running: kernel_splitter\n");

    #pragma omp parallel
    {
    #if USE_LIKWID
    likwid_markerThreadInit();
    likwid_markerStartRegion("splitter");
    #endif
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
                #pragma ivdep
                for (int x = 0; x < SUBGRIDSIZE; x++) {
                    // Compute shifted position in subgrid
                    int x_dst = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                    int y_dst = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

                    #if ORDER == ORDER_BL_V_U_P
                    (*subgrid)[bl][chunk][y_dst][x_dst][0] = (*grid)[0][grid_y+y][grid_x+x];
                    (*subgrid)[bl][chunk][y_dst][x_dst][1] = (*grid)[1][grid_y+y][grid_x+x];
                    (*subgrid)[bl][chunk][y_dst][x_dst][2] = (*grid)[2][grid_y+y][grid_x+x];
                    (*subgrid)[bl][chunk][y_dst][x_dst][3] = (*grid)[3][grid_y+y][grid_x+x];
                    #elif ORDER == ORDER_BL_P_V_U
                    (*subgrid)[bl][chunk][0][y_dst][x_dst] = (*grid)[0][grid_y+y][grid_x+x];
                    (*subgrid)[bl][chunk][1][y_dst][x_dst] = (*grid)[1][grid_y+y][grid_x+x];
                    (*subgrid)[bl][chunk][2][y_dst][x_dst] = (*grid)[2][grid_y+y][grid_x+x];
                    (*subgrid)[bl][chunk][3][y_dst][x_dst] = (*grid)[3][grid_y+y][grid_x+x];
                    #endif
                }
            }
        }
    }
    #if USE_LIKWID
    likwid_markerStopRegion("splitter");
    #endif
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
