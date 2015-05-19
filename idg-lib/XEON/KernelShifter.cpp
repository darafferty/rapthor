#include <complex.h>
#include <stdint.h>

#include "Types.h"


extern "C" {
void kernel_shifter(
	const int jobsize,
	SubGridType __restrict__ *subgrid
	) {
    #pragma omp parallel
    {
    #if USE_LIKWID
    likwid_markerThreadInit();
    likwid_markerStartRegion("shifter");
    #endif
    #pragma omp for
    for (int bl = 0; bl < jobsize; bl++) {
        for (int chunk = 0; chunk < CHUNKSIZE; chunk++) {
            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                float complex _subgrid[SUBGRIDSIZE][SUBGRIDSIZE];
                
                // Make copy of uvgrid
                for (int y = 0; y < SUBGRIDSIZE; y++) {
		            for (int x = 0; x < SUBGRIDSIZE; x++) {
		                #if ORDER == ORDER_BL_V_U_P
                        _subgrid[y][x] = (*subgrid)[bl][chunk][y][x][pol];
                        #elif ORDER == ORDER_BL_P_V_U
                        _subgrid[y][x] = (*subgrid)[bl][chunk][pol][y][x];
                        #endif
		            }
	            }
	            
	            // Update uv grid
	            #pragma unroll_and_jam(2)
                for (int y = 0; y < SUBGRIDSIZE; y++) {
		            for (int x = 0; x < SUBGRIDSIZE; x++) {
                        int x_dst = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
		                int y_dst = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
		                #if ORDER == ORDER_BL_V_U_P
                        (*subgrid)[bl][chunk][y_dst][x_dst][pol] = _subgrid[y][x];
                        #elif ORDER == ORDER_BL_P_V_U
                        (*subgrid)[bl][chunk][pol][y_dst][x_dst] = _subgrid[y][x];
                        #endif
                    }
                }	        
            }
        }
    }
    #if USE_LIKWID
    likwid_markerStopRegion("shifter");
    #endif
    }
}

uint64_t kernel_shifter_flops(int jobsize) {
    return 1ULL * jobsize * SUBGRIDSIZE * SUBGRIDSIZE * 6;
}

uint64_t kernel_shifter_bytes(int jobsize) {
    return 1ULL * jobsize * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * 3 * sizeof(float complex);
}
}
