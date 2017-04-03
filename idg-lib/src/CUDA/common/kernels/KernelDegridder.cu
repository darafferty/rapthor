#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

#define BATCH_SIZE DEGRIDDER_BATCH_SIZE
#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))


extern "C" {
__global__ void kernel_degridder(
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
          float2*        __restrict__ subgrid
    ) {
    int tidx       = threadIdx.x;
    int tidy       = threadIdx.y;
    int tid        = tidx + tidy * blockDim.y;
    int nr_threads = blockDim.x * blockDim.y;
    int s          = blockIdx.x;

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
    const float w_offset_in_lambda = w_step * ((float)m.coordinate.z + 0.5);
    const float w_offset = 2*M_PI*w_offset_in_lambda;

    // Compute u and v offset in wavelenghts
    float u_offset = (x_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
    float v_offset = (y_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;

    // Shared data
    __shared__ float4 _pix[NR_POLARIZATIONS / 2][BATCH_SIZE];
    __shared__ float4 _lmn_phaseoffset[BATCH_SIZE];

    __syncthreads();

    // Prepare pixels
    const int nr_pixels = subgrid_size * subgrid_size;
    for (int i = tid; i < nr_pixels; i += nr_threads) {
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
        float2 aXX2 = cuConjf(aterm[station2_idx + 0]);
        float2 aXY2 = cuConjf(aterm[station2_idx + 1]);
        float2 aYX2 = cuConjf(aterm[station2_idx + 2]);
        float2 aYY2 = cuConjf(aterm[station2_idx + 3]);

        // Load spheroidal
        float spheroidal_ = spheroidal[y * subgrid_size + x];

        // Compute shifted position in subgrid
        int x_src = (x + (subgrid_size/2)) % subgrid_size;
        int y_src = (y + (subgrid_size/2)) % subgrid_size;

        // Load uv values
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

    // Iterate all visibilities
    for (int i = tid; i < ALIGN(nr_timesteps * nr_channels, nr_threads); i += nr_threads) {
        int time = i / nr_channels;
        int chan = i % nr_channels;

        float2 visXX, visXY, visYX, visYY;
        float  u, v, w;
        float  wavenumber;

        if (time < nr_timesteps) {
            visXX = make_float2(0, 0);
            visXY = make_float2(0, 0);
            visYX = make_float2(0, 0);
            visYY = make_float2(0, 0);

            u = uvw[time_offset_global + time].u;
            v = uvw[time_offset_global + time].v;
            w = uvw[time_offset_global + time].w;

            wavenumber = wavenumbers[chan];
        }

        __syncthreads();

        // Iterate all pixels
        int current_nr_pixels = BATCH_SIZE;
        for (int pixel_offset = 0; pixel_offset < nr_pixels; pixel_offset += current_nr_pixels) {
            current_nr_pixels = nr_pixels - pixel_offset < BATCH_SIZE ?
                                nr_pixels - pixel_offset : BATCH_SIZE;

            __syncthreads();

            // Prepare data
            for (int i = tid; i < current_nr_pixels; i += nr_threads) {
                int y = (pixel_offset + i) / subgrid_size;
                int x = (pixel_offset + i) % subgrid_size;

                // Compute shifted position in subgrid
                int x_src = (x + (subgrid_size/2)) % subgrid_size;
                int y_src = (y + (subgrid_size/2)) % subgrid_size;

                // Load pixels from device memory
                float2 pixelsXX = subgrid[index_subgrid(subgrid_size, s, 0, y_src, x_src)];
                float2 pixelsXY = subgrid[index_subgrid(subgrid_size, s, 1, y_src, x_src)];
                float2 pixelsYX = subgrid[index_subgrid(subgrid_size, s, 2, y_src, x_src)];
                float2 pixelsYY = subgrid[index_subgrid(subgrid_size, s, 3, y_src, x_src)];

                // Store pixels
                _pix[0][i] = make_float4(pixelsXX.x, pixelsXX.y, pixelsXY.x, pixelsXY.y);
                _pix[1][i] = make_float4(pixelsYX.x, pixelsYX.y, pixelsYY.x, pixelsYY.y);

                // Compute l,m,n and phase offset
                const float l = (x+0.5-(subgrid_size/2)) * image_size/subgrid_size;
                const float m = (y+0.5-(subgrid_size/2)) * image_size/subgrid_size;
                const float tmp = (l * l) + (m * m);
                const float n = tmp / (1.0f + sqrtf(1.0f - tmp));
                float phase_offset = u_offset*l + v_offset*m + w_offset*n;
                _lmn_phaseoffset[i] = make_float4(l, m, n, phase_offset);
            }

             __syncthreads();

            // Iterate batch
            for (int j = 0; j < current_nr_pixels; j++) {
                // Load l,m,n
                float l = _lmn_phaseoffset[j].x;
                float m = _lmn_phaseoffset[j].y;
                float n = _lmn_phaseoffset[j].z;

                // Load phase offset
                float phase_offset = _lmn_phaseoffset[j].w;

                // Compute phase index
                float phase_index = u * l + v * m + w * n;

                // Compute phasor
                float  phase  = (phase_index * wavenumber) - phase_offset;
                float2 phasor = make_float2(cosf(phase), sinf(phase));

                // Load pixels from shared memory
                float2 apXX = make_float2(_pix[0][j].x, _pix[0][j].y);
                float2 apXY = make_float2(_pix[0][j].z, _pix[0][j].w);
                float2 apYX = make_float2(_pix[1][j].x, _pix[1][j].y);
                float2 apYY = make_float2(_pix[1][j].z, _pix[1][j].w);

                // Multiply pixels by phasor
                visXX.x += phasor.x * apXX.x;
                visXX.y += phasor.x * apXX.y;
                visXX.x -= phasor.y * apXX.y;
                visXX.y += phasor.y * apXX.x;

                visXY.x += phasor.x * apXY.x;
                visXY.y += phasor.x * apXY.y;
                visXY.x -= phasor.y * apXY.y;
                visXY.y += phasor.y * apXY.x;

                visYX.x += phasor.x * apYX.x;
                visYX.y += phasor.x * apYX.y;
                visYX.x -= phasor.y * apYX.y;
                visYX.y += phasor.y * apYX.x;

                visYY.x += phasor.x * apYY.x;
                visYY.y += phasor.x * apYY.y;
                visYY.x -= phasor.y * apYY.y;
                visYY.y += phasor.y * apYY.x;
            } // end for j (batch)

        } // end for pixels

        // Store visibility
        const float scale = 1.0f / (nr_pixels);
        int idx_time = time_offset_global + time;
        int idx_xx = index_visibility(nr_channels, idx_time, chan, 0);
        int idx_xy = index_visibility(nr_channels, idx_time, chan, 1);
        int idx_yx = index_visibility(nr_channels, idx_time, chan, 2);
        int idx_yy = index_visibility(nr_channels, idx_time, chan, 3);

        if (time < nr_timesteps) {
            visibilities[idx_xx] = visXX * scale;
            visibilities[idx_xy] = visXY * scale;
            visibilities[idx_yx] = visYX * scale;
            visibilities[idx_yy] = visYY * scale;
        }
    } // end for i (visibilities)

    __syncthreads();
} // end kernel_degridder
} // end extern "C"
