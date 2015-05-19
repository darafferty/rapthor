#include <stdio.h>
#include <stdlib.h>

#include <idg/Common/Types.h>

extern "C" {

/*
	Kernel
*/
void kernel_splitter(
    const int jobsize,
    const CoordinateType __restrict__ *coordinates,
    UVGridType           __restrict__ *uvgrid,
    const GridType       __restrict__ *grid
    ) {
    #pragma omp parallel for
    for (int bl = 0; bl < jobsize; bl++) {
        for (int y = 0; y < BLOCKSIZE; y++) {
            #pragma ivdep
            for (int x = 0; x < BLOCKSIZE; x++) {
                int uv_x = (*coordinates)[bl].x;
                int uv_y = (*coordinates)[bl].y;

                #if ORDER == ORDER_BL_V_U_P
                (*uvgrid)[bl][y][x][0] = (*grid)[0][uv_y+y][uv_x+x];
                (*uvgrid)[bl][y][x][1] = (*grid)[1][uv_y+y][uv_x+x];
                (*uvgrid)[bl][y][x][2] = (*grid)[2][uv_y+y][uv_x+x];
                (*uvgrid)[bl][y][x][3] = (*grid)[3][uv_y+y][uv_x+x];
                #elif ORDER == ORDER_BL_P_V_U
                (*uvgrid)[bl][0][y][x] = (*grid)[0][uv_y+y][uv_x+x];
                (*uvgrid)[bl][1][y][x] = (*grid)[1][uv_y+y][uv_x+x];
                (*uvgrid)[bl][2][y][x] = (*grid)[2][uv_y+y][uv_x+x];
                (*uvgrid)[bl][3][y][x] = (*grid)[3][uv_y+y][uv_x+x];
                #endif
            }
        }
    }
}

#include <idg/Common/Parameters.h>

}
