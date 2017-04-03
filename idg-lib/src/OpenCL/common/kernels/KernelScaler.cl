#include "math.cl"

#include "Types.cl"

/*
	Kernel
*/
__kernel void kernel_scaler(
    __global float2* subgrid)
{
	int tidx = get_local_id(0);
	int tidy = get_local_id(1);
    int tid = tidx + tidy * get_local_size(0);
    int blocksize = get_local_size(0) * get_local_size(1);
    int s = get_group_id(0);

    // Compute scaling factor
    const float scale = 1.0 / ((float) SUBGRIDSIZE * (float) SUBGRIDSIZE);

	// Iterate all pixels in subgrid
    for (int i = tid; i < SUBGRIDSIZE * SUBGRIDSIZE; i += blocksize) {
        int y = i / SUBGRIDSIZE;
        int x = i % SUBGRIDSIZE;

        for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
            int idx = index_subgrid(SUBGRIDSIZE, s, pol, y, x);
            float2 value = subgrid[idx];
            subgrid[idx] = (float2) (value.x * scale, value.y * scale);
        }
    }
}
