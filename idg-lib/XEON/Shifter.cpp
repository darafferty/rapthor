#include <idg/Common/Types.h>

extern "C" {

/*
	Kernel
*/
void kernel_shifter(
	const int jobsize,
	UVGridType __restrict__ *uvgrid
	) {
    
    #pragma omp parallel for
    for (int bl = 0; bl < jobsize; bl++) {
        for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
            std::complex<float> _uvgrid[BLOCKSIZE][BLOCKSIZE];
            
            // Make copy of uvgrid
            for (int y = 0; y < BLOCKSIZE; y++) {
		        for (int x = 0; x < BLOCKSIZE; x++) {
		            #if ORDER == ORDER_BL_V_U_P
                    _uvgrid[y][x] = (*uvgrid)[bl][y][x][pol];
                    #elif ORDER == ORDER_BL_P_V_U
                    _uvgrid[y][x] = (*uvgrid)[bl][pol][y][x];
                    #endif
		        }
	        }
	        
	        // Update uv grid
	        #pragma unroll_and_jam(2)
            for (int y = 0; y < BLOCKSIZE; y++) {
		        for (int x = 0; x < BLOCKSIZE; x++) {
                    int x_dst = (x + (BLOCKSIZE/2)) % BLOCKSIZE;
		            int y_dst = (y + (BLOCKSIZE/2)) % BLOCKSIZE;
		            #if ORDER == ORDER_BL_V_U_P
                    (*uvgrid)[bl][y_dst][x_dst][pol] = _uvgrid[y][x];
                    #elif ORDER == ORDER_BL_P_V_U
                    (*uvgrid)[bl][pol][y_dst][x_dst] = _uvgrid[y][x];
                    #endif
                }
            }	        
        }
    }
}

#include <idg/Common/Parameters.h>

}
