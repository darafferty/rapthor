#include "math.cl"

#include "Types.cl"

/*
	Kernel
*/
__kernel void kernel_scaler(
    __global SubGridType subgrid
	) {
    int s = get_global_id(0);
	int tidx = get_local_id(0);
	int tidy = get_local_id(1);

    // Compute scaling factor
    float scale = 1 / ((float) SUBGRIDSIZE * (float) SUBGRIDSIZE);

	// Iterate all pixels in subgrid
	for (int y = tidy; y < SUBGRIDSIZE; y += get_local_size(1)) {
		for (int x = tidx; x < SUBGRIDSIZE; x += get_local_size(0)) {
            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                float2 value = subgrid[s][pol][y][x];
                subgrid[s][pol][y][x] = (float2) (value.x * scale, value.y * scale);
            }
        }
    }
}
