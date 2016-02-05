#include "math.cl"

#include "Types.cl"

/*
	Kernel
*/
__kernel void kernel_scaler(
    __global SubGridType subgrid
	) {
	int tidx = get_local_id(0);
	int tidy = get_local_id(1);
    int tid = tidx + tidy * get_local_size(0);
    int blocksize = get_local_size(0) * get_local_size(1);
    int s = get_group_id(0);

    // Compute scaling factor
    const float scale = 1.0 / (SUBGRIDSIZE * SUBGRIDSIZE);

	// Iterate all pixels in subgrid
    for (int i = tid; i < SUBGRIDSIZE * SUBGRIDSIZE; i += blocksize) {
        for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
            float2 value = subgrid[s][pol][0][i];
            subgrid[s][pol][0][i] = (float2) (value.x * scale, value.y * scale);
        }
    }
}
