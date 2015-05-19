#include <stdio.h>
#include <stdlib.h>

#include <idg/Common/Types.h>

extern "C" {

/*
	Kernel
*/
void kernel_adder(
    const int jobsize,
    const CoordinateType __restrict__ *coordinates,
    const UVGridType     __restrict__ *uvgrid,
    GridType             __restrict__ *grid
    ) {
    #pragma omp parallel for
    for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
        for (int bl = 0; bl < jobsize; bl++) {
            for (int y = 0; y < BLOCKSIZE; y++) {
                #pragma ivdep
                for (int x = 0; x < BLOCKSIZE; x++) {
                    int uv_x = (*coordinates)[bl].x;
                    int uv_y = (*coordinates)[bl].y;

                    #if ORDER == ORDER_BL_V_U_P
                    (*grid)[pol][uv_y+y][uv_x+x] += (*uvgrid)[bl][y][x][pol];
                    #elif ORDER == ORDER_BL_P_V_U
                    (*grid)[pol][uv_y+y][uv_x+x] += (*uvgrid)[bl][pol][y][x];
                    #endif
                }
            }
        }
    }
}

#include <idg/Common/Parameters.h>

}
