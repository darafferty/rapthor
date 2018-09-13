#include "Types.h"
#include "math.cu"

#define BATCH_SIZE DEGRIDDER_BATCH_SIZE
#define BLOCK_SIZE DEGRIDDER_BLOCK_SIZE
#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))

#define MAX_NR_CHANNELS 8

__shared__ float4 shared[3][BATCH_SIZE];

/*
    Kernel
*/
template<int current_nr_channels>
__device__ void kernel_degridder_(
    const int                         grid_size,
    const int                         subgrid_size,
    const float                       image_size,
    const float                       w_step,
    const int                         nr_channels,
    const int                         channel_offset,
    const int                         nr_stations,
    const UVW*           __restrict__ uvw,
    const float*         __restrict__ wavenumbers,
          float2*        __restrict__ visibilities,
    const float*         __restrict__ spheroidal,
    const float2*        __restrict__ aterm,
    const Metadata*      __restrict__ metadata,
          float2*        __restrict__ subgrid)
{
    int s          = blockIdx.x;
    int tidx       = threadIdx.x;
    int tidy       = threadIdx.y;
    int tid        = tidx + tidy * blockDim.x;
    int nr_threads = blockDim.x * blockDim.y;

    // Load metadata for first subgrid
    const Metadata &m_0 = metadata[0];

    // Load metadata for current subgrid
    const Metadata &m = metadata[s];
    const int time_offset_global = (m.baseline_offset - m_0.baseline_offset) + m.time_offset;
    const int nr_timesteps = m.nr_timesteps;
    const int aterm_index = m.aterm_index;
    const int station1 = m.baseline.station1;
    const int station2 = m.baseline.station2;
    const int x_coordinate = m.coordinate.x;
    const int y_coordinate = m.coordinate.y;

    // Compute u,v,w offset in wavelenghts
    const float u_offset = (x_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
    const float v_offset = (y_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
    const float w_offset = w_step * ((float) m.coordinate.z + 0.5) * 2 * M_PI;

    if (channel_offset == 0) {
        // Apply spheroidal and aterm correction
        for (int i = tid; i < subgrid_size * subgrid_size; i += nr_threads) {
            int y = i / subgrid_size;
            int x = i % subgrid_size;

            // Load aterm for station1
            int station1_idx = index_aterm(subgrid_size, nr_stations, aterm_index, station1, y, x);
            float2 aXX1 = aterm[station1_idx + 0];
            float2 aXY1 = aterm[station1_idx + 1];
            float2 aYX1 = aterm[station1_idx + 2];
            float2 aYY1 = aterm[station1_idx + 3];

            // Load aterm for station2
            int station2_idx = index_aterm(subgrid_size, nr_stations, aterm_index, station2, y, x);
            float2 aXX2 = aterm[station2_idx + 0];
            float2 aXY2 = aterm[station2_idx + 1];
            float2 aYX2 = aterm[station2_idx + 2];
            float2 aYY2 = aterm[station2_idx + 3];

            // Load spheroidal
            float spheroidal_ = spheroidal[y * subgrid_size + x];

            // Compute shifted position in subgrid
            int x_src = (x + (subgrid_size/2)) % subgrid_size;
            int y_src = (y + (subgrid_size/2)) % subgrid_size;

            // Load pixels
            int idx_xx = index_subgrid(subgrid_size, s, 0, y_src, x_src);
            int idx_xy = index_subgrid(subgrid_size, s, 1, y_src, x_src);
            int idx_yx = index_subgrid(subgrid_size, s, 2, y_src, x_src);
            int idx_yy = index_subgrid(subgrid_size, s, 3, y_src, x_src);
            float2 pixelsXX = spheroidal_ * subgrid[idx_xx];
            float2 pixelsXY = spheroidal_ * subgrid[idx_xy];
            float2 pixelsYX = spheroidal_ * subgrid[idx_yx];
            float2 pixelsYY = spheroidal_ * subgrid[idx_yy];

            // Apply aterm
            apply_aterm(
                aXX1, aXY1, aYX1, aYY1,
                aXX2, aXY2, aYX2, aYY2,
                pixelsXX, pixelsXY, pixelsYX, pixelsYY);

            // Store pixels
            subgrid[idx_xx] = pixelsXX;
            subgrid[idx_xy] = pixelsXY;
            subgrid[idx_yx] = pixelsYX;
            subgrid[idx_yy] = pixelsYY;
        }
    }

    // Iterate visibilities
    for (int time = tid; time < ALIGN(nr_timesteps, nr_threads); time += nr_threads) {
        float2 visXX[current_nr_channels];
        float2 visXY[current_nr_channels];
        float2 visYX[current_nr_channels];
        float2 visYY[current_nr_channels];

        for (int chan = 0; chan < current_nr_channels; chan++) {
            visXX[chan] = make_float2(0, 0);
            visXY[chan] = make_float2(0, 0);
            visYX[chan] = make_float2(0, 0);
            visYY[chan] = make_float2(0, 0);
        }

        float u, v, w;

        if (time < nr_timesteps) {
            u = uvw[time_offset_global + time].u;
            v = uvw[time_offset_global + time].v;
            w = uvw[time_offset_global + time].w;
        }

        __syncthreads();

        // Iterate pixels
        const int nr_pixels = subgrid_size * subgrid_size;
        int current_nr_pixels = BATCH_SIZE;
        for (int pixel_offset = 0; pixel_offset < nr_pixels; pixel_offset += current_nr_pixels) {
            current_nr_pixels = nr_pixels - pixel_offset < min(nr_threads, BATCH_SIZE) ?
                                nr_pixels - pixel_offset : min(nr_threads, BATCH_SIZE);

            __syncthreads();

            // Prepare data
            for (int j = tid; j < current_nr_pixels; j += nr_threads) {
                int y = (pixel_offset + j) / subgrid_size;
                int x = (pixel_offset + j) % subgrid_size;

                // Compute shifted position in subgrid
                int x_src = (x + (subgrid_size/2)) % subgrid_size;
                int y_src = (y + (subgrid_size/2)) % subgrid_size;

                // Load pixels
                int idx_xx = index_subgrid(subgrid_size, s, 0, y_src, x_src);
                int idx_xy = index_subgrid(subgrid_size, s, 1, y_src, x_src);
                int idx_yx = index_subgrid(subgrid_size, s, 2, y_src, x_src);
                int idx_yy = index_subgrid(subgrid_size, s, 3, y_src, x_src);
                float2 pixelsXX = subgrid[idx_xx];
                float2 pixelsXY = subgrid[idx_xy];
                float2 pixelsYX = subgrid[idx_yx];
                float2 pixelsYY = subgrid[idx_yy];

                // Compute l,m,n and phase offset
                const float l = compute_l(x, subgrid_size, image_size);
                const float m = compute_m(y, subgrid_size, image_size);
                const float n = compute_n(l, m);
                float phase_offset = u_offset*l + v_offset*m + w_offset*n;

                // Store values in shared memory
                shared[0][j] = make_float4(pixelsXX.x, pixelsXX.y, pixelsXY.x, pixelsXY.y);
                shared[1][j] = make_float4(pixelsYX.x, pixelsYX.y, pixelsYY.x, pixelsYY.y);
                shared[2][j] = make_float4(l, m, n, phase_offset);
            } // end for j (pixels)

             __syncthreads();

            // Iterate current batch of pixels
            for (int k = 0; k < current_nr_pixels; k++) {
                // Load pixels from shared memory
                float2 apXX = make_float2(shared[0][k].x, shared[0][k].y);
                float2 apXY = make_float2(shared[0][k].z, shared[0][k].w);
                float2 apYX = make_float2(shared[1][k].x, shared[1][k].y);
                float2 apYY = make_float2(shared[1][k].z, shared[1][k].w);

                // Load l,m,n
                float l = shared[2][k].x;
                float m = shared[2][k].y;
                float n = shared[2][k].z;

                // Load phase offset
                float phase_offset = shared[2][k].w;

                // Compute phase index
                float phase_index = u * l + v * m + w * n;

                for (int chan = 0; chan < current_nr_channels; chan++) {
                    // Load wavenumber
                    float wavenumber = wavenumbers[channel_offset + chan];

                    // Compute phasor
                    float  phase  = (phase_index * wavenumber) - phase_offset;
                    float2 phasor = make_float2(cosf(phase), sinf(phase));

                    // Multiply pixels by phasor
                    visXX[chan].x += phasor.x * apXX.x;
                    visXX[chan].y += phasor.x * apXX.y;
                    visXX[chan].x -= phasor.y * apXX.y;
                    visXX[chan].y += phasor.y * apXX.x;

                    visXY[chan].x += phasor.x * apXY.x;
                    visXY[chan].y += phasor.x * apXY.y;
                    visXY[chan].x -= phasor.y * apXY.y;
                    visXY[chan].y += phasor.y * apXY.x;

                    visYX[chan].x += phasor.x * apYX.x;
                    visYX[chan].y += phasor.x * apYX.y;
                    visYX[chan].x -= phasor.y * apYX.y;
                    visYX[chan].y += phasor.y * apYX.x;

                    visYY[chan].x += phasor.x * apYY.x;
                    visYY[chan].y += phasor.x * apYY.y;
                    visYY[chan].x -= phasor.y * apYY.y;
                    visYY[chan].y += phasor.y * apYY.x;
                } // end for chan
            } // end for k (batch)
        } // end for j (pixels)

        for (int chan = 0; chan < current_nr_channels; chan++) {
            // Store visibility
            const float scale = 1.0f / (nr_pixels);
            int idx_time = time_offset_global + time;
            int idx_chan = channel_offset + chan;
            int idx_xx = index_visibility(nr_channels, idx_time, idx_chan, 0);
            int idx_xy = index_visibility(nr_channels, idx_time, idx_chan, 1);
            int idx_yx = index_visibility(nr_channels, idx_time, idx_chan, 2);
            int idx_yy = index_visibility(nr_channels, idx_time, idx_chan, 3);

            if (time < nr_timesteps) {
                visibilities[idx_xx] = visXX[chan] * scale;
                visibilities[idx_xy] = visXY[chan] * scale;
                visibilities[idx_yx] = visYX[chan] * scale;
                visibilities[idx_yy] = visYY[chan] * scale;
            }
        } // end for chan
    } // end for time
} // end kernel_degridder_

#define KERNEL_DEGRIDDER_TEMPLATE(current_nr_channels) \
    for (; (channel_offset + current_nr_channels) <= nr_channels; channel_offset += current_nr_channels) { \
        kernel_degridder_<current_nr_channels>( \
            grid_size, subgrid_size, image_size, w_step, nr_channels, channel_offset, nr_stations, \
            uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrid); \
    }

extern "C" {
__global__ void
__launch_bounds__(BLOCK_SIZE)
    kernel_degridder(
    const int                         grid_size,
    const int                         subgrid_size,
    const float                       image_size,
    const float                       w_step,
    const int                         nr_channels,
    const int                         nr_stations,
    const UVW*           __restrict__ uvw,
    const float*         __restrict__ wavenumbers,
          float2*        __restrict__ visibilities,
    const float*         __restrict__ spheroidal,
    const float2*        __restrict__ aterm,
    const Metadata*      __restrict__ metadata,
          float2*        __restrict__ subgrid)
{
	int channel_offset = 0;
	assert(MAX_NR_CHANNELS == 8);
	KERNEL_DEGRIDDER_TEMPLATE(8);
	KERNEL_DEGRIDDER_TEMPLATE(4);
	KERNEL_DEGRIDDER_TEMPLATE(2);
	KERNEL_DEGRIDDER_TEMPLATE(1);
}

} // end extern "C"
