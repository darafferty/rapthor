#include "math.cl"

#include "Types.cl"

#define MAX_NR_TIMESTEPS GRIDDER_BATCH_SIZE

#define NR_CHANNELS_8  8
#define NR_CHANNELS_4  4

/*
	Kernel
*/
__kernel void kernel_gridder_1(
    const int gridsize,
    const float imagesize,
    const float w_step,
    const int nr_channels,
    const int channel_offset,
    const int nr_stations,
    __global const UVWType			uvw,
    __global const WavenumberType	wavenumbers,
    __global const VisibilitiesType	visibilities,
    __global const SpheroidalType	spheroidal,
    __global const ATermType		aterm,
    __global const MetadataType		metadata,
    __global       SubGridType      subgrid,
    __local        float4           _visibilities[MAX_NR_TIMESTEPS][NR_POLARIZATIONS/2],
    __local        float4           _uvw[MAX_NR_TIMESTEPS]
	) {
	int tidx = get_local_id(0);
	int tidy = get_local_id(1);
	int tid = tidx + tidy * get_local_size(0);;
    int blocksize = get_local_size(0) * get_local_size(1);
    int s = get_group_id(0);

    // Set subgrid to zero
    if (channel_offset == 0) {
        for (int i = tid; i < SUBGRIDSIZE * SUBGRIDSIZE; i += blocksize) {
            subgrid[s][0][0][i] = (float2) (0, 0);
            subgrid[s][1][0][i] = (float2) (0, 0);
            subgrid[s][2][0][i] = (float2) (0, 0);
            subgrid[s][3][0][i] = (float2) (0, 0);
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // Load metadata for first subgrid
    const Metadata m_0 = metadata[0];

    // Load metadata for current subgrid
	const Metadata m = metadata[s];
    const int time_offset_global = (m.baseline_offset - m_0.baseline_offset) + (m.time_offset - m_0.time_offset);
    const int nr_timesteps = m.nr_timesteps;
	const int aterm_index = m.aterm_index;
	const int station1 = m.baseline.station1;
	const int station2 = m.baseline.station2;
	const int x_coordinate = m.coordinate.x;
	const int y_coordinate = m.coordinate.y;
    const float w_offset = w_step * m.coordinate.z;

    // Iterate all timesteps
    int current_nr_timesteps = MAX_NR_TIMESTEPS;
    for (int time_offset_local = 0; time_offset_local < nr_timesteps; time_offset_local += current_nr_timesteps) {
        current_nr_timesteps = nr_timesteps - time_offset_local < MAX_NR_TIMESTEPS ?
                               nr_timesteps - time_offset_local : MAX_NR_TIMESTEPS;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Load UVW
        for (int time = tid; time < current_nr_timesteps; time += blocksize) {
            UVW a = uvw[time_offset_global + time_offset_local + time];
            _uvw[time] = (float4) (a.u, a.v, a.w, 0);
        }

        // Load visibilities
	    for (int i = tid; i < current_nr_timesteps; i += blocksize) {
            int time = i;
            int index = (time_offset_global + time_offset_local + time) * nr_channels + channel_offset;
            float2 a = visibilities[index][0];
            float2 b = visibilities[index][1];
            float2 c = visibilities[index][2];
            float2 d = visibilities[index][3];
            _visibilities[i][0] = (float4) (a.x, a.y, b.x, b.y);
            _visibilities[i][1] = (float4) (c.x, c.y, d.x, d.y);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute u and v offset in wavelenghts
        float u_offset = (x_coordinate + SUBGRIDSIZE/2 -gridsize/2) / imagesize * 2 * M_PI;
        float v_offset = (y_coordinate + SUBGRIDSIZE/2 -gridsize/2) / imagesize * 2 * M_PI;

        // Iterate all pixels in subgrid
        for (int i = tid; i < SUBGRIDSIZE * SUBGRIDSIZE; i += blocksize) {
            int y = i / SUBGRIDSIZE;
            int x = i % SUBGRIDSIZE;

            // Private subgrid points
            float2 uvXX = (float2) (0, 0);
            float2 uvXY = (float2) (0, 0);
            float2 uvYX = (float2) (0, 0);
            float2 uvYY = (float2) (0, 0);

            // Compute l,m,n
            float l = (x+0.5-(SUBGRIDSIZE/2)) * imagesize/SUBGRIDSIZE;
            float m = (y+0.5-(SUBGRIDSIZE/2)) * imagesize/SUBGRIDSIZE;
            float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));

            // Iterate all timesteps
            for (int time = 0; time < current_nr_timesteps; time++) {
                 // Load UVW coordinates
                float u = _uvw[time].x;
                float v = _uvw[time].y;
                float w = _uvw[time].z;

                // Compute phase index
                float phase_index = u*l + v*m + w*n;

                // Compute phase offset
                float phase_offset = u_offset*l + v_offset*m + w_offset*n;

                // Compute phasor
                float wavenumber = wavenumbers[channel_offset];
                float phase = phase_offset - (phase_index * wavenumber);
                float2 phasor = (float2) (native_cos(phase), native_sin(phase));

                // Load visibilities from shared memory
                float4 a = _visibilities[time][0];
                float4 b = _visibilities[time][1];
                float2 visXX = (float2) (a.x, a.y);
                float2 visXY = (float2) (a.z, a.w);
                float2 visYX = (float2) (b.x, b.y);
                float2 visYY = (float2) (b.z, b.w);

                // Multiply visibility by phasor
                uvXX.x += phasor.x * visXX.x;
                uvXX.y += phasor.x * visXX.y;
                uvXX.x -= phasor.y * visXX.y;
                uvXX.y += phasor.y * visXX.x;

                uvXY.x += phasor.x * visXY.x;
                uvXY.y += phasor.x * visXY.y;
                uvXY.x -= phasor.y * visXY.y;
                uvXY.y += phasor.y * visXY.x;

                uvYX.x += phasor.x * visYX.x;
                uvYX.y += phasor.x * visYX.y;
                uvYX.x -= phasor.y * visYX.y;
                uvYX.y += phasor.y * visYX.x;

                uvYY.x += phasor.x * visYY.x;
                uvYY.y += phasor.x * visYY.y;
                uvYY.x -= phasor.y * visYY.y;
                uvYY.y += phasor.y * visYY.x;
            }

            // Get a term for station1
            float2 aXX1 = aterm[aterm_index * nr_stations + station1][y][x][0];
            float2 aXY1 = aterm[aterm_index * nr_stations + station1][y][x][1];
            float2 aYX1 = aterm[aterm_index * nr_stations + station1][y][x][2];
            float2 aYY1 = aterm[aterm_index * nr_stations + station1][y][x][3];

            // Get aterm for station2
            float2 aXX2 = conj(aterm[aterm_index * nr_stations + station2][y][x][0]);
            float2 aXY2 = conj(aterm[aterm_index * nr_stations + station2][y][x][1]);
            float2 aYX2 = conj(aterm[aterm_index * nr_stations + station2][y][x][2]);
            float2 aYY2 = conj(aterm[aterm_index * nr_stations + station2][y][x][3]);

            // Apply aterm
            apply_aterm(
                aXX1,   aXY1,  aYX1,  aYY1,
                aXX2,   aXY2,  aYX2,  aYY2,
                &uvXX, &uvXY, &uvYX, &uvYY);

            // Load spheroidal
            float sph = spheroidal[y][x];

            // Compute shifted position in subgrid
            int x_dst = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
            int y_dst = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

            // Apply spheroidal and update uv grid
            subgrid[s][0][y_dst][x_dst] += uvXX * sph;
            subgrid[s][1][y_dst][x_dst] += uvXY * sph;
            subgrid[s][2][y_dst][x_dst] += uvYX * sph;
            subgrid[s][3][y_dst][x_dst] += uvYY * sph;
        }
    }
}

__kernel void kernel_gridder_4(
    const int gridsize,
    const float imagesize,
    const float w_step,
    const int nr_channels,
    const int channel_offset,
    const int nr_stations,
    __global const UVWType			uvw,
    __global const WavenumberType	wavenumbers,
    __global const VisibilitiesType	visibilities,
    __global const SpheroidalType	spheroidal,
    __global const ATermType		aterm,
    __global const MetadataType		metadata,
    __global       SubGridType      subgrid,
    __local        float4           _visibilities[MAX_NR_TIMESTEPS][NR_CHANNELS_4][NR_POLARIZATIONS/2],
    __local        float4           _uvw[MAX_NR_TIMESTEPS],
    __local        float            _wavenumbers[NR_CHANNELS_4]
	) {
	int tidx = get_local_id(0);
	int tidy = get_local_id(1);
	int tid = tidx + tidy * get_local_size(0);;
    int blocksize = get_local_size(0) * get_local_size(1);
    int s = get_group_id(0);

    // Set subgrid to zero
    if (channel_offset == 0) {
        for (int i = tid; i < SUBGRIDSIZE * SUBGRIDSIZE; i += blocksize) {
            subgrid[s][0][0][i] = (float2) (0, 0);
            subgrid[s][1][0][i] = (float2) (0, 0);
            subgrid[s][2][0][i] = (float2) (0, 0);
            subgrid[s][3][0][i] = (float2) (0, 0);
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // Load metadata for first subgrid
    const Metadata m_0 = metadata[0];

    // Load metadata for current subgrid
	const Metadata m = metadata[s];
    const int time_offset_global = (m.baseline_offset - m_0.baseline_offset) + (m.time_offset - m_0.time_offset);
    const int nr_timesteps = m.nr_timesteps;
	const int aterm_index = m.aterm_index;
	const int station1 = m.baseline.station1;
	const int station2 = m.baseline.station2;
	const int x_coordinate = m.coordinate.x;
	const int y_coordinate = m.coordinate.y;
    const float w_offset = w_step * m.coordinate.z;

    // Load wavenumbers
    for (int i = tid; i < NR_CHANNELS_4; i += blocksize) {
        _wavenumbers[i] = wavenumbers[channel_offset + i];
    }

    // Iterate all timesteps
    int current_nr_timesteps = MAX_NR_TIMESTEPS;
    for (int time_offset_local = 0; time_offset_local < nr_timesteps; time_offset_local += current_nr_timesteps) {
        current_nr_timesteps = nr_timesteps - time_offset_local < MAX_NR_TIMESTEPS ?
                               nr_timesteps - time_offset_local : MAX_NR_TIMESTEPS;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Load UVW
        for (int time = tid; time < current_nr_timesteps; time += blocksize) {
            UVW a = uvw[time_offset_global + time_offset_local + time];
            _uvw[time] = (float4) (a.u, a.v, a.w, 0);
        }

        // Load visibilities
	    for (int i = tid; i < current_nr_timesteps * NR_CHANNELS_4; i += blocksize) {
            int time = i / NR_CHANNELS_4;
            int chan = i % NR_CHANNELS_4;
            int index = (time_offset_global + time_offset_local + time) * nr_channels + (channel_offset + chan);
            float2 a = visibilities[index][0];
            float2 b = visibilities[index][1];
            float2 c = visibilities[index][2];
            float2 d = visibilities[index][3];
            _visibilities[0][i][0] = (float4) (a.x, a.y, b.x, b.y);
            _visibilities[0][i][1] = (float4) (c.x, c.y, d.x, d.y);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute u and v offset in wavelenghts
        float u_offset = (x_coordinate + SUBGRIDSIZE/2 -gridsize/2) / imagesize * 2 * M_PI;
        float v_offset = (y_coordinate + SUBGRIDSIZE/2 -gridsize/2) / imagesize * 2 * M_PI;

        // Iterate all pixels in subgrid
        for (int i = tid; i < SUBGRIDSIZE * SUBGRIDSIZE; i += blocksize) {
            int y = i / SUBGRIDSIZE;
            int x = i % SUBGRIDSIZE;

            // Private subgrid points
            float2 uvXX = (float2) (0, 0);
            float2 uvXY = (float2) (0, 0);
            float2 uvYX = (float2) (0, 0);
            float2 uvYY = (float2) (0, 0);

            // Compute l,m,n
            float l = (x+0.5-(SUBGRIDSIZE/2)) * imagesize/SUBGRIDSIZE;
            float m = (y+0.5-(SUBGRIDSIZE/2)) * imagesize/SUBGRIDSIZE;
            float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));

            // Iterate all timesteps
            for (int time = 0; time < current_nr_timesteps; time++) {
                 // Load UVW coordinates
                float u = _uvw[time].x;
                float v = _uvw[time].y;
                float w = _uvw[time].z;

                // Compute phase index
                float phase_index = u*l + v*m + w*n;

                // Compute phase offset
                float phase_offset = u_offset*l + v_offset*m + w_offset*n;

                // Compute phasor
                #pragma unroll
                for (int chan = 0; chan < NR_CHANNELS_4; chan++) {
                    float wavenumber = _wavenumbers[chan];
                    float phase = phase_offset - (phase_index * wavenumber);
                    float2 phasor = (float2) (native_cos(phase), native_sin(phase));

                    // Load visibilities from shared memory
                    float4 a = _visibilities[time][chan][0];
                    float4 b = _visibilities[time][chan][1];
                    float2 visXX = (float2) (a.x, a.y);
                    float2 visXY = (float2) (a.z, a.w);
                    float2 visYX = (float2) (b.x, b.y);
                    float2 visYY = (float2) (b.z, b.w);

                    // Multiply visibility by phasor
                    uvXX.x += phasor.x * visXX.x;
                    uvXX.y += phasor.x * visXX.y;
                    uvXX.x -= phasor.y * visXX.y;
                    uvXX.y += phasor.y * visXX.x;

                    uvXY.x += phasor.x * visXY.x;
                    uvXY.y += phasor.x * visXY.y;
                    uvXY.x -= phasor.y * visXY.y;
                    uvXY.y += phasor.y * visXY.x;

                    uvYX.x += phasor.x * visYX.x;
                    uvYX.y += phasor.x * visYX.y;
                    uvYX.x -= phasor.y * visYX.y;
                    uvYX.y += phasor.y * visYX.x;

                    uvYY.x += phasor.x * visYY.x;
                    uvYY.y += phasor.x * visYY.y;
                    uvYY.x -= phasor.y * visYY.y;
                    uvYY.y += phasor.y * visYY.x;
                }
            }

            // Get a term for station1
            float2 aXX1 = aterm[aterm_index * nr_stations + station1][y][x][0];
            float2 aXY1 = aterm[aterm_index * nr_stations + station1][y][x][1];
            float2 aYX1 = aterm[aterm_index * nr_stations + station1][y][x][2];
            float2 aYY1 = aterm[aterm_index * nr_stations + station1][y][x][3];

            // Get aterm for station2
            float2 aXX2 = conj(aterm[aterm_index * nr_stations + station2][y][x][0]);
            float2 aXY2 = conj(aterm[aterm_index * nr_stations + station2][y][x][1]);
            float2 aYX2 = conj(aterm[aterm_index * nr_stations + station2][y][x][2]);
            float2 aYY2 = conj(aterm[aterm_index * nr_stations + station2][y][x][3]);

            // Apply aterm
            apply_aterm(
                aXX1,   aXY1,  aYX1,  aYY1,
                aXX2,   aXY2,  aYX2,  aYY2,
                &uvXX, &uvXY, &uvYX, &uvYY);

            // Load spheroidal
            float sph = spheroidal[y][x];

            // Compute shifted position in subgrid
            int x_dst = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
            int y_dst = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

            // Apply spheroidal and update uv grid
            subgrid[s][0][y_dst][x_dst] += uvXX * sph;
            subgrid[s][1][y_dst][x_dst] += uvXY * sph;
            subgrid[s][2][y_dst][x_dst] += uvYX * sph;
            subgrid[s][3][y_dst][x_dst] += uvYY * sph;
        }
    }
}

__kernel void kernel_gridder_8(
    const int gridsize,
    const float imagesize,
    const float w_step,
    const int nr_channels,
    const int channel_offset,
    const int nr_stations,
    __global const UVWType			uvw,
    __global const WavenumberType	wavenumbers,
    __global const VisibilitiesType	visibilities,
    __global const SpheroidalType	spheroidal,
    __global const ATermType		aterm,
    __global const MetadataType		metadata,
    __global       SubGridType      subgrid,
    __local        float4           _visibilities[MAX_NR_TIMESTEPS][NR_CHANNELS_8][NR_POLARIZATIONS/2],
    __local        float4           _uvw[MAX_NR_TIMESTEPS],
    __local        float            _wavenumbers[NR_CHANNELS_8]
	) {
	int tidx = get_local_id(0);
	int tidy = get_local_id(1);
	int tid = tidx + tidy * get_local_size(0);;
    int blocksize = get_local_size(0) * get_local_size(1);
    int s = get_group_id(0);

    // Set subgrid to zero
    if (channel_offset == 0) {
        for (int i = tid; i < SUBGRIDSIZE * SUBGRIDSIZE; i += blocksize) {
            subgrid[s][0][0][i] = (float2) (0, 0);
            subgrid[s][1][0][i] = (float2) (0, 0);
            subgrid[s][2][0][i] = (float2) (0, 0);
            subgrid[s][3][0][i] = (float2) (0, 0);
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // Load metadata for first subgrid
    const Metadata m_0 = metadata[0];

    // Load metadata for current subgrid
	const Metadata m = metadata[s];
    const int time_offset_global = (m.baseline_offset - m_0.baseline_offset) + (m.time_offset - m_0.time_offset);
    const int nr_timesteps = m.nr_timesteps;
	const int aterm_index = m.aterm_index;
	const int station1 = m.baseline.station1;
	const int station2 = m.baseline.station2;
	const int x_coordinate = m.coordinate.x;
	const int y_coordinate = m.coordinate.y;
    const float w_offset = w_step * m.coordinate.z;

    // Load wavenumbers
    for (int i = tid; i < NR_CHANNELS_8; i += blocksize) {
        _wavenumbers[i] = wavenumbers[channel_offset + i];
    }

    // Iterate all timesteps
    int current_nr_timesteps = MAX_NR_TIMESTEPS;
    for (int time_offset_local = 0; time_offset_local < nr_timesteps; time_offset_local += current_nr_timesteps) {
        current_nr_timesteps = nr_timesteps - time_offset_local < MAX_NR_TIMESTEPS ?
                               nr_timesteps - time_offset_local : MAX_NR_TIMESTEPS;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Load UVW
        for (int time = tid; time < current_nr_timesteps; time += blocksize) {
            UVW a = uvw[time_offset_global + time_offset_local + time];
            _uvw[time] = (float4) (a.u, a.v, a.w, 0);
        }

        // Load visibilities
	    for (int i = tid; i < current_nr_timesteps * NR_CHANNELS_8; i += blocksize) {
            int time = i / NR_CHANNELS_8;
            int chan = i % NR_CHANNELS_8;
            int index = (time_offset_global + time_offset_local + time) * nr_channels + (channel_offset + chan);
            float2 a = visibilities[index][0];
            float2 b = visibilities[index][1];
            float2 c = visibilities[index][2];
            float2 d = visibilities[index][3];
            _visibilities[0][i][0] = (float4) (a.x, a.y, b.x, b.y);
            _visibilities[0][i][1] = (float4) (c.x, c.y, d.x, d.y);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute u and v offset in wavelenghts
        float u_offset = (x_coordinate + SUBGRIDSIZE/2 -gridsize/2) / imagesize * 2 * M_PI;
        float v_offset = (y_coordinate + SUBGRIDSIZE/2 -gridsize/2) / imagesize * 2 * M_PI;

        // Iterate all pixels in subgrid
        for (int i = tid; i < SUBGRIDSIZE * SUBGRIDSIZE; i += blocksize) {
            int y = i / SUBGRIDSIZE;
            int x = i % SUBGRIDSIZE;

            // Private subgrid points
            float2 uvXX = (float2) (0, 0);
            float2 uvXY = (float2) (0, 0);
            float2 uvYX = (float2) (0, 0);
            float2 uvYY = (float2) (0, 0);

            // Compute l,m,n
            float l = (x+0.5-(SUBGRIDSIZE/2)) * imagesize/SUBGRIDSIZE;
            float m = (y+0.5-(SUBGRIDSIZE/2)) * imagesize/SUBGRIDSIZE;
            float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));

            // Iterate all timesteps
            for (int time = 0; time < current_nr_timesteps; time++) {
                 // Load UVW coordinates
                float u = _uvw[time].x;
                float v = _uvw[time].y;
                float w = _uvw[time].z;

                // Compute phase index
                float phase_index = u*l + v*m + w*n;

                // Compute phase offset
                float phase_offset = u_offset*l + v_offset*m + w_offset*n;

                // Compute phasor
                #pragma unroll
                for (int chan = 0; chan < NR_CHANNELS_8; chan++) {
                    float wavenumber = _wavenumbers[chan];
                    float phase = phase_offset - (phase_index * wavenumber);
                    float2 phasor = (float2) (native_cos(phase), native_sin(phase));

                    // Load visibilities from shared memory
                    float4 a = _visibilities[time][chan][0];
                    float4 b = _visibilities[time][chan][1];
                    float2 visXX = (float2) (a.x, a.y);
                    float2 visXY = (float2) (a.z, a.w);
                    float2 visYX = (float2) (b.x, b.y);
                    float2 visYY = (float2) (b.z, b.w);

                    // Multiply visibility by phasor
                    uvXX.x += phasor.x * visXX.x;
                    uvXX.y += phasor.x * visXX.y;
                    uvXX.x -= phasor.y * visXX.y;
                    uvXX.y += phasor.y * visXX.x;

                    uvXY.x += phasor.x * visXY.x;
                    uvXY.y += phasor.x * visXY.y;
                    uvXY.x -= phasor.y * visXY.y;
                    uvXY.y += phasor.y * visXY.x;

                    uvYX.x += phasor.x * visYX.x;
                    uvYX.y += phasor.x * visYX.y;
                    uvYX.x -= phasor.y * visYX.y;
                    uvYX.y += phasor.y * visYX.x;

                    uvYY.x += phasor.x * visYY.x;
                    uvYY.y += phasor.x * visYY.y;
                    uvYY.x -= phasor.y * visYY.y;
                    uvYY.y += phasor.y * visYY.x;
                }
            }

            // Get a term for station1
            float2 aXX1 = aterm[aterm_index * nr_stations + station1][y][x][0];
            float2 aXY1 = aterm[aterm_index * nr_stations + station1][y][x][1];
            float2 aYX1 = aterm[aterm_index * nr_stations + station1][y][x][2];
            float2 aYY1 = aterm[aterm_index * nr_stations + station1][y][x][3];

            // Get aterm for station2
            float2 aXX2 = conj(aterm[aterm_index * nr_stations + station2][y][x][0]);
            float2 aXY2 = conj(aterm[aterm_index * nr_stations + station2][y][x][1]);
            float2 aYX2 = conj(aterm[aterm_index * nr_stations + station2][y][x][2]);
            float2 aYY2 = conj(aterm[aterm_index * nr_stations + station2][y][x][3]);

            // Apply aterm
            apply_aterm(
                aXX1,   aXY1,  aYX1,  aYY1,
                aXX2,   aXY2,  aYX2,  aYY2,
                &uvXX, &uvXY, &uvYX, &uvYY);

            // Load spheroidal
            float sph = spheroidal[y][x];

            // Compute shifted position in subgrid
            int x_dst = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
            int y_dst = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

            // Apply spheroidal and update uv grid
            subgrid[s][0][y_dst][x_dst] += uvXX * sph;
            subgrid[s][1][y_dst][x_dst] += uvXY * sph;
            subgrid[s][2][y_dst][x_dst] += uvYX * sph;
            subgrid[s][3][y_dst][x_dst] += uvYY * sph;
        }
    }
}

__kernel void kernel_gridder(
    const int gridsize,
    const float imagesize,
	const float w_step,
    const int nr_channels,
    const int nr_stations,
	__global const UVWType			uvw,
	__global const WavenumberType	wavenumbers,
	__global const VisibilitiesType	visibilities,
	__global const SpheroidalType	spheroidal,
	__global const ATermType		aterm,
	__global const MetadataType		metadata,
	__global       SubGridType		subgrid
	) {
    int channel_offset = 0;

    for (; (channel_offset + 8) <= nr_channels; channel_offset += 8) {
        __local float4 _visibilities[MAX_NR_TIMESTEPS][NR_CHANNELS_8][NR_POLARIZATIONS/2];
        __local float4 _uvw[MAX_NR_TIMESTEPS];
        __local float  _wavenumbers[NR_CHANNELS_8];

        kernel_gridder_8(
            gridsize, imagesize, w_step, nr_channels, channel_offset, nr_stations,
            uvw, wavenumbers, visibilities,spheroidal, aterm, metadata, subgrid,
            _visibilities, _uvw, _wavenumbers);
    }

    for (; (channel_offset + 4) <= nr_channels; channel_offset += 4) {
        __local float4 _visibilities[MAX_NR_TIMESTEPS][NR_CHANNELS_4][NR_POLARIZATIONS/2];
        __local float4 _uvw[MAX_NR_TIMESTEPS];
        __local float  _wavenumbers[NR_CHANNELS_4];

        kernel_gridder_4(
            gridsize, imagesize, w_step, nr_channels, channel_offset, nr_stations,
            uvw, wavenumbers, visibilities,spheroidal, aterm, metadata, subgrid,
            _visibilities, _uvw, _wavenumbers);
    }


    for (; channel_offset < nr_channels; channel_offset++) {
	    __local float4 _visibilities[MAX_NR_TIMESTEPS][NR_POLARIZATIONS/2];
	    __local float4 _uvw[MAX_NR_TIMESTEPS];

        kernel_gridder_1(
            gridsize, imagesize, w_step, nr_channels, channel_offset, nr_stations,
            uvw, wavenumbers, visibilities,spheroidal, aterm, metadata, subgrid,
            _visibilities, _uvw);
    }
}

