#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

#define BATCH_SIZE GRIDDER_BATCH_SIZE

/*
    Kernel
*/
template<int current_nr_channels>
__device__ void kernel_gridder_(
    const int                    grid_size,
    const int                    subgrid_size,
    const float                  image_size,
    const float                  w_offset,
    const int                    nr_channels,
    const int                    channel_offset,
    const int                    nr_stations,
    const UVW*      __restrict__ uvw,
    const float*    __restrict__ wavenumbers,
    const float2*   __restrict__ visibilities,
    const float*    __restrict__ spheroidal,
    const float2*   __restrict__ aterm,
    const Metadata* __restrict__ metadata,
          float2*   __restrict__ subgrid
    ) {
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tid = tidx + tidy * blockDim.x;
    int blockSize = blockDim.x * blockDim.y;
    int s = blockIdx.x;

    // Set subgrid to zero
    if (channel_offset == 0) {
        for (int i = tid; i < subgrid_size * subgrid_size; i += blockSize) {
            int idx_xx = index_subgrid(subgrid_size, s, 0, 0, i);
            int idx_xy = index_subgrid(subgrid_size, s, 1, 0, i);
            int idx_yx = index_subgrid(subgrid_size, s, 2, 0, i);
            int idx_yy = index_subgrid(subgrid_size, s, 3, 0, i);
            subgrid[idx_xx] = make_float2(0, 0);
            subgrid[idx_xy] = make_float2(0, 0);
            subgrid[idx_yx] = make_float2(0, 0);
            subgrid[idx_yy] = make_float2(0, 0);
        }
    }

    __syncthreads();

    // Load metadata for first subgrid
    const Metadata &m_0 = metadata[0];

    // Load metadata for current subgrid
    const Metadata &m = metadata[s];
    const int time_offset_global = (m.baseline_offset - m_0.baseline_offset) + (m.time_offset - m_0.time_offset);
    const int nr_timesteps = m.nr_timesteps;
    const int aterm_index = m.aterm_index;
    const int station1 = m.baseline.station1;
    const int station2 = m.baseline.station2;
    const int x_coordinate = m.coordinate.x;
    const int y_coordinate = m.coordinate.y;

    // Shared data
    __shared__ float4 _visibilities[2][BATCH_SIZE][current_nr_channels];
    __shared__ float4 _uvw[BATCH_SIZE];
    __shared__ float _wavenumbers[current_nr_channels];

    // Load wavenumbers
    for (int i = tid; i < current_nr_channels; i += blockSize) {
        _wavenumbers[i] = wavenumbers[channel_offset + i];
    }

    // Iterate all timesteps
    int current_nr_timesteps = BATCH_SIZE;
    for (int time_offset_local = 0; time_offset_local < nr_timesteps; time_offset_local += current_nr_timesteps) {
        current_nr_timesteps = nr_timesteps - time_offset_local < BATCH_SIZE ?
                               nr_timesteps - time_offset_local : BATCH_SIZE;

        __syncthreads();

        // Load UVW
        for (int time = tid; time < current_nr_timesteps; time += blockSize) {
            UVW a = uvw[time_offset_global + time_offset_local + time];
            _uvw[time] = make_float4(a.u, a.v, a.w, 0);
        }

        // Load visibilities
        for (int i = tid; i < current_nr_timesteps * current_nr_channels; i += blockSize) {
            int time = i / current_nr_channels;
            int chan = i % current_nr_channels;
            int idx_time = time_offset_global + time_offset_local + time;
            int idx_chan = channel_offset + chan;
            int idx_xx = index_visibility(nr_channels, idx_time, idx_chan, 0);
            int idx_xy = index_visibility(nr_channels, idx_time, idx_chan, 1);
            int idx_yx = index_visibility(nr_channels, idx_time, idx_chan, 2);
            int idx_yy = index_visibility(nr_channels, idx_time, idx_chan, 3);
            float2 a = visibilities[idx_xx];
            float2 b = visibilities[idx_xy];
            float2 c = visibilities[idx_yx];
            float2 d = visibilities[idx_yy];
            _visibilities[0][time][chan] = make_float4(a.x, a.y, b.x, b.y);
            _visibilities[1][time][chan] = make_float4(c.x, c.y, d.x, d.y);
        }

        __syncthreads();

        // Compute u and v offset in wavelenghts
        float u_offset = (x_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
        float v_offset = (y_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;

        // Iterate all pixels in subgrid
        for (int i = tid; i < subgrid_size * subgrid_size; i += blockSize) {
            int y = i / subgrid_size;
            int x = i % subgrid_size;

            // Private pixels
            float2 uvXX = make_float2(0, 0);
            float2 uvXY = make_float2(0, 0);
            float2 uvYX = make_float2(0, 0);
            float2 uvYY = make_float2(0, 0);

            // Compute l,m,n
            float l = (x+0.5-(subgrid_size/2)) * image_size/subgrid_size;
            float m = (y+0.5-(subgrid_size/2)) * image_size/subgrid_size;
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
                #pragma unroll current_nr_channels
                for (int chan = 0; chan < current_nr_channels; chan++) {
                    float wavenumber = _wavenumbers[chan];
                    float phase = phase_offset - (phase_index * wavenumber);
                    float2 phasor = make_float2(cos(phase), sin(phase));

                    // Load visibilities from shared memory
                    float4 a = _visibilities[0][time][chan];
                    float4 b = _visibilities[1][time][chan];
                    float2 visXX = make_float2(a.x, a.y);
                    float2 visXY = make_float2(a.z, a.w);
                    float2 visYX = make_float2(b.x, b.y);
                    float2 visYY = make_float2(b.z, b.w);

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
            float2 aXX2 = cuConjf(aterm[station2_idx + 0]);
            float2 aXY2 = cuConjf(aterm[station2_idx + 1]);
            float2 aYX2 = cuConjf(aterm[station2_idx + 2]);
            float2 aYY2 = cuConjf(aterm[station2_idx + 3]);

            // Apply aterm
            apply_aterm(
                aXX1, aXY1, aYX1, aYY1,
                aXX2, aXY2, aYX2, aYY2,
                uvXX, uvXY, uvYX, uvYY);

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
}

#define KERNEL_GRIDDER_TEMPLATE(NR_CHANNELS) \
    for (; (channel_offset + NR_CHANNELS) <= nr_channels; channel_offset += NR_CHANNELS) { \
        kernel_gridder_<NR_CHANNELS>( \
            grid_size, subgrid_size, image_size, w_offset, nr_channels, channel_offset, nr_stations, \
            uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrid); \
    }

extern "C" {
__global__ void kernel_gridder(
    const int                           grid_size,
    const int                           subgrid_size,
    const float                         image_size,
    const float                         w_offset,
    const int                           nr_channels,
    const int                           nr_stations,
    const UVW*             __restrict__ uvw,
    const float*           __restrict__ wavenumbers,
    const float2*          __restrict__ visibilities,
    const float*           __restrict__ spheroidal,
    const float2*          __restrict__ aterm,
    const Metadata*        __restrict__ metadata,
          float2*          __restrict__ subgrid
    ) {
    int channel_offset = 0;
    KERNEL_GRIDDER_TEMPLATE(8)
    KERNEL_GRIDDER_TEMPLATE(4)
    KERNEL_GRIDDER_TEMPLATE(3)
    KERNEL_GRIDDER_TEMPLATE(2)
    KERNEL_GRIDDER_TEMPLATE(1)
}
}
