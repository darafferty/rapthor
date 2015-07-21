#include "math.cl"

#include "Types.cl"


/*
	Kernel
*/
__kernel void kernel_splitter(
    const int bl_offset,
	__global const UVWType  uvw,
	__global SubGridType	subgrid,
	__global const GridType grid
	) {
	int tid = get_local_id(0);
	int bl = get_global_id(0) / get_local_size(0);

    // Precompute grid coordinates for every subgrid
    __local Coordinate _coordinates[NR_CHUNKS];
    for (int chunk = tid; chunk < NR_CHUNKS; chunk += get_local_size(0)) {
        // Load UVW coordinates for chunk
        int time_offset = chunk * CHUNKSIZE;
        UVW uvw_first = uvw[bl][time_offset];
        UVW uvw_last  = uvw[bl][time_offset + CHUNKSIZE - 1];
       
        // Compute grid coordinate for chunk 
        int grid_x = ((uvw_first.u + uvw_last.u) / 2) - (SUBGRIDSIZE / 2);
        int grid_y = ((uvw_first.v + uvw_last.v) / 2) - (SUBGRIDSIZE / 2);

        // Store grid coordinate
        _coordinates[chunk].x = grid_x;
        _coordinates[chunk].y = grid_y;
    }
    barrier(CLK_LOCAL_MEM_FENCE); 

    // Extract all subgrids from grid
    for (int chunk = 0; chunk < NR_CHUNKS; chunk++) {
        for (int pixel = tid; pixel < SUBGRIDSIZE * SUBGRIDSIZE; pixel += get_local_size(0)) {
            // Compute x and y position
            int x = pixel % SUBGRIDSIZE;
            int y = pixel / SUBGRIDSIZE;
            
            // Compute thread position in grid
            int grid_x = _coordinates[chunk].x + x;
            int grid_y = _coordinates[chunk].y + y;
            
            // Add subgrid point to grid
            #if ORDER == ORDER_BL_P_V_U
            atomicAdd(&(subgrid[bl][chunk][0][y][x]), grid[0][grid_y][grid_x]);
            atomicAdd(&(subgrid[bl][chunk][1][y][x]), grid[1][grid_y][grid_x]);
            atomicAdd(&(subgrid[bl][chunk][2][y][x]), grid[2][grid_y][grid_x]);
            atomicAdd(&(subgrid[bl][chunk][3][y][x]), grid[3][grid_y][grid_x]);
            #elif ORDER == ORDER_BL_V_U_P
            atomicAdd(&(subgrid[bl][chunk][y][x][0]), grid[0][grid_y][grid_x]);
            atomicAdd(&(subgrid[bl][chunk][y][x][1]), grid[1][grid_y][grid_x]);
            atomicAdd(&(subgrid[bl][chunk][y][x][2]), grid[2][grid_y][grid_x]);
            atomicAdd(&(subgrid[bl][chunk][y][x][3]), grid[3][grid_y][grid_x]);
            #endif
        }

        barrier(CLK_LOCAL_MEM_FENCE); 
    }
}
