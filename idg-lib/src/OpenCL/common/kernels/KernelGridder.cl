#include "math.cl"

#include "Types.cl"

#define BATCH_SIZE GRIDDER_BATCH_SIZE
#define BLOCK_SIZE GRIDDER_BLOCK_SIZE
#define MAX_NR_CHANNELS 8

/*
    Kernel
*/
__kernel
__attribute__((work_group_size_hint(BLOCK_SIZE, 1, 1)))
void kernel_gridder_(
    const int                current_nr_channels,
    const int                grid_size,
    const int                subgrid_size,
    const float              image_size,
    const float              w_step,
    const int                nr_channels,
    const int                channel_offset,
    const int                nr_stations,
    __global const UVW*      uvw,
    __global const float*    wavenumbers,
    __global const float2*   visibilities,
    __global const float*    spheroidal,
    __global const float2*   aterm,
    __global const Metadata* metadata,
    __global       float2*   subgrid,
    __local float4           visibilities_[BATCH_SIZE][MAX_NR_CHANNELS][NR_POLARIZATIONS/2],
    __local float4           uvw_[BATCH_SIZE],
    __local float            wavenumbers_[MAX_NR_CHANNELS])
{
    int tidx = get_local_id(0);
    int tidy = get_local_id(1);
    int tid = tidx + tidy * get_local_size(0);;
    int nr_threads = get_local_size(0) * get_local_size(1);
    int s = get_group_id(0);

    // Set subgrid to zero
    if (channel_offset == 0) {
        for (int i = tid; i < subgrid_size * subgrid_size; i += nr_threads) {
            int idx_xx = index_subgrid(subgrid_size, s, 0, 0, i);
            int idx_xy = index_subgrid(subgrid_size, s, 1, 0, i);
            int idx_yx = index_subgrid(subgrid_size, s, 2, 0, i);
            int idx_yy = index_subgrid(subgrid_size, s, 3, 0, i);
            subgrid[idx_xx] = (float2) (0, 0);
            subgrid[idx_xy] = (float2) (0, 0);
            subgrid[idx_yx] = (float2) (0, 0);
            subgrid[idx_yy] = (float2) (0, 0);
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // Load metadata for first subgrid
    const Metadata m_0 = metadata[0];

    // Load metadata for current subgrid
    const Metadata m = metadata[s];
    const int time_offset_global = (m.baseline_offset - m_0.baseline_offset) + m.time_offset;
    const int nr_timesteps = m.nr_timesteps;
    const int aterm_index = m.aterm_index;
    const int station1 = m.baseline.station1;
    const int station2 = m.baseline.station2;
    const int x_coordinate = m.coordinate.x;
    const int y_coordinate = m.coordinate.y;
    const float w_offset_in_lambda = w_step * ((float)m.coordinate.z + 0.5);
    const float w_offset = 2*M_PI*w_offset_in_lambda;

    // Load wavenumbers
    for (int i = tid; i < current_nr_channels; i += nr_threads) {
        wavenumbers_[i] = wavenumbers[channel_offset + i];
    }

    // Iterate all timesteps
    int current_nr_timesteps = BATCH_SIZE;
    for (int time_offset_local = 0; time_offset_local < nr_timesteps; time_offset_local += current_nr_timesteps) {
        current_nr_timesteps = nr_timesteps - time_offset_local < BATCH_SIZE ?
                               nr_timesteps - time_offset_local : BATCH_SIZE;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Load UVW
        for (int time = tid; time < current_nr_timesteps; time += nr_threads) {
            UVW a = uvw[time_offset_global + time_offset_local + time];
            uvw_[time] = (float4) (a.u, a.v, a.w, 0);
        }

        // Load visibilities
        for (int i = tid; i < current_nr_timesteps * current_nr_channels; i += nr_threads) {
            int time = i / current_nr_channels;
            int chan = i % current_nr_channels;
            if (time < current_nr_timesteps && chan < current_nr_channels) {
                int idx_time = time_offset_global + time_offset_local + time;
                int idx_chan = channel_offset + chan;
                int idx_vis = index_visibility(nr_channels, idx_time, idx_chan);
                float2 a = visibilities[idx_vis + 0];
                float2 b = visibilities[idx_vis + 1];
                float2 c = visibilities[idx_vis + 2];
                float2 d = visibilities[idx_vis + 3];
                visibilities_[0][i][0] = (float4) (a.x, a.y, b.x, b.y);
                visibilities_[0][i][1] = (float4) (c.x, c.y, d.x, d.y);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute u and v offset in wavelenghts
        float u_offset = (x_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
        float v_offset = (y_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;

        // Iterate all pixels in subgrid
        for (int i = tid; i < subgrid_size * subgrid_size; i += nr_threads) {
            int y = i / subgrid_size;
            int x = i % subgrid_size;

            // Private pixels
            float2 uvXX = (float2) (0, 0);
            float2 uvXY = (float2) (0, 0);
            float2 uvYX = (float2) (0, 0);
            float2 uvYY = (float2) (0, 0);

            // Compute l,m,n
            float l = (x+0.5-(subgrid_size/2)) * image_size/subgrid_size;
            float m = (y+0.5-(subgrid_size/2)) * image_size/subgrid_size;
            float tmp = (l * l) + (m * m);
            float n = tmp / (1.0f + native_sqrt(1.0f - tmp));

            // Iterate all timesteps
            for (int time = 0; time < current_nr_timesteps; time++) {
                // Load UVW coordinates
                float u = uvw_[time].x;
                float v = uvw_[time].y;
                float w = uvw_[time].z;

                // Compute phase index
                float phase_index = u*l + v*m + w*n;

                // Compute phase offset
                float phase_offset = u_offset*l + v_offset*m + w_offset*n;

                // Compute phasor
                #pragma unroll
                for (int chan = 0; chan < current_nr_channels; chan++) {
                    float wavenumber = wavenumbers_[chan];
                    float phase = phase_offset - (phase_index * wavenumber);
                    float2 phasor = (float2) (native_cos(phase), native_sin(phase));

                    // Load visibilities from shared memory
                    float4 a = visibilities_[time][chan][0];
                    float4 b = visibilities_[time][chan][1];
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
            } // end for time

            // Get aterm for station1
            int station1_idx = index_aterm(subgrid_size, nr_stations, aterm_index, station1, y, x);
            float2 aXX1 = aterm[station1_idx + 0];
            float2 aXY1 = aterm[station1_idx + 1];
            float2 aYX1 = aterm[station1_idx + 2];
            float2 aYY1 = aterm[station1_idx + 3];

            // Get aterm for station2
            int station2_idx = index_aterm(subgrid_size, nr_stations, aterm_index, station2, y, x);
            float2 aXX2 = conj(aterm[station2_idx + 0]);
            float2 aXY2 = conj(aterm[station2_idx + 1]);
            float2 aYX2 = conj(aterm[station2_idx + 2]);
            float2 aYY2 = conj(aterm[station2_idx + 3]);

            // Apply aterm
            apply_aterm(
                aXX1,   aXY1,  aYX1,  aYY1,
                aXX2,   aXY2,  aYX2,  aYY2,
                &uvXX, &uvXY, &uvYX, &uvYY);

            // Load spheroidal
            float spheroidal_ = spheroidal[y * subgrid_size + x];

            // Compute shifted position in subgrid
            int x_dst = (x + (subgrid_size/2)) % subgrid_size;
            int y_dst = (y + (subgrid_size/2)) % subgrid_size;

            // Set subgrid value
            int idx_xx = index_subgrid(subgrid_size, s, 0, y_dst, x_dst);
            int idx_xy = index_subgrid(subgrid_size, s, 1, y_dst, x_dst);
            int idx_yx = index_subgrid(subgrid_size, s, 2, y_dst, x_dst);
            int idx_yy = index_subgrid(subgrid_size, s, 3, y_dst, x_dst);
            subgrid[idx_xx] += uvXX * spheroidal_;
            subgrid[idx_xy] += uvXY * spheroidal_;
            subgrid[idx_yx] += uvYX * spheroidal_;
            subgrid[idx_yy] += uvYY * spheroidal_;
        } // end for i
    } // end for time_offset_local
} // end kernel_gridder_

#define KERNEL_GRIDDER_TEMPLATE(NR_CHANNELS) \
    for (; (channel_offset + NR_CHANNELS) <= nr_channels; channel_offset += NR_CHANNELS) { \
        kernel_gridder_( \
            NR_CHANNELS, grid_size, subgrid_size, image_size, w_step, nr_channels, channel_offset, nr_stations, \
            uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrid, \
            visibilities_, uvw_, wavenumbers_); \
    }

__kernel void kernel_gridder(
    const int                grid_size,
    const int                subgrid_size,
    const float              image_size,
    const float              w_step,
    const int                nr_channels,
    const int                nr_stations,
    __global const UVW*      uvw,
    __global const float*    wavenumbers,
    __global const float2*   visibilities,
    __global const float*    spheroidal,
    __global const float2*   aterm,
    __global const Metadata* metadata,
    __global       float2*   subgrid)
{
    __local float4 visibilities_[BATCH_SIZE][MAX_NR_CHANNELS][NR_POLARIZATIONS/2];
    __local float4 uvw_[BATCH_SIZE];
    __local float  wavenumbers_[MAX_NR_CHANNELS];

    int channel_offset = 0;
    for (int i = 8; i > 0; i--) {
        KERNEL_GRIDDER_TEMPLATE(i);
    }
} // end kernel_gridder

