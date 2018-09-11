#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))

#define MAX_NR_CHANNELS 8

/*
    Kernel
*/
__kernel
__attribute__((work_group_size_hint(BLOCK_SIZE, 1, 1)))
void kernel_degridder(
    const int                grid_size,
    const int                subgrid_size,
    const float              image_size,
    const float              w_step,
    const int                nr_stations,
    __global const UVW*      uvw,
    __global const float*    wavenumbers,
    __global       float2*   visibilities,
    __global const float*    spheroidal,
    __global const float2*   aterm,
    __global const Metadata* metadata,
    __global float2*         subgrid)
{
    int s = get_group_id(0);
    int tidx = get_local_id(0);
    int tidy = get_local_id(1);
    int tid  = tidx + tidy * get_local_size(0);
    int nr_threads = get_local_size(0) * get_local_size(1);

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

    // Compute u,v,w offset in wavelenghts
    const float u_offset = (x_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
    const float v_offset = (y_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
    const float w_offset = w_step * ((float)m.coordinate.z + 0.5) * 2 * M_PI;

    // Shared data
    __local float8 pixels_[BATCH_SIZE];
    __local float4 lmn_phaseoffset_[BATCH_SIZE];

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
        float8 aterm1 = (float8) (aXX1, aYY1, aXX1, aYY1);
        float8 aterm2 = (float8) (aYX1, aXY1, aYX1, aXY1);

        // Load aterm for station2
        int station2_idx = index_aterm(subgrid_size, nr_stations, aterm_index, station2, y, x);
        float2 aXX2 = conj(aterm[station2_idx + 0]);
        float2 aXY2 = conj(aterm[station2_idx + 1]);
        float2 aYX2 = conj(aterm[station2_idx + 2]);
        float2 aYY2 = conj(aterm[station2_idx + 3]);
        float8 aterm3 = (float8) (aXX2, aXX2, aYY2, aYY2);
        float8 aterm4 = (float8) (aYX2, aYX2, aXY2, aXY2);

        // Compute shifted position in subgrid
        int x_src = (x + (subgrid_size/2)) % subgrid_size;
        int y_src = (y + (subgrid_size/2)) % subgrid_size;

        // Load pixels
        int idx_xx = index_subgrid(subgrid_size, s, 0, y_src, x_src);
        int idx_xy = index_subgrid(subgrid_size, s, 1, y_src, x_src);
        int idx_yx = index_subgrid(subgrid_size, s, 2, y_src, x_src);
        int idx_yy = index_subgrid(subgrid_size, s, 3, y_src, x_src);
        float8 pixels = (float8) (
                    subgrid[idx_xx], subgrid[idx_xy],
                    subgrid[idx_yx], subgrid[idx_yy]);

        // Apply spheroidal
        pixels *= spheroidal[y * subgrid_size + x];

        float8 pixels_aterm = (float8) 0;

        // Apply aterm to pixels: P*A1
        // [ uvXX, uvXY;    [ aXX1, aXY1;
        //   uvYX, uvYY ] *   aYX1, aYY1 ]
        pixels_aterm += cmul8(pixels, aterm1);
        pixels_aterm += cmul8(pixels, aterm2);

        // Apply aterm to pixels: A2^H*P
        // [ aXX2, aYX1;      [ uvXX, uvXY;
        //   aXY1, aYY2 ]  *    uvYX, uvYY ]
        pixels = pixels_aterm;
        pixels_aterm = (float8) (0);
        pixels_aterm += cmul8(pixels, aterm3);
        pixels_aterm += cmul8(pixels, aterm4);

        // Store pixels
        subgrid[idx_xx] = pixels.S01;
        subgrid[idx_xy] = pixels.S23;
        subgrid[idx_yx] = pixels.S45;
        subgrid[idx_yy] = pixels.S67;
    }

    int current_nr_channels = MAX_NR_CHANNELS;
    for (int channel_offset = 0; channel_offset < NR_CHANNELS; channel_offset += MAX_NR_CHANNELS) {
        current_nr_channels = NR_CHANNELS - channel_offset < min(nr_threads, MAX_NR_CHANNELS) ?
                              NR_CHANNELS - channel_offset : min(nr_threads, MAX_NR_CHANNELS);

        // Iterate timesteps
        for (int time = tid; time < ALIGN(nr_timesteps, nr_threads); time += nr_threads) {
            float8 visibility[MAX_NR_CHANNELS];

            for (int chan = 0; chan < MAX_NR_CHANNELS; chan++) {
                visibility[chan] = (float8) 0;
            }

            float4 uvw_ = (float4) 0;

            if (time < nr_timesteps) {
                uvw_.x = uvw[time_offset_global + time].u;
                uvw_.y = uvw[time_offset_global + time].v;
                uvw_.z = uvw[time_offset_global + time].w;
            }

            // Iterate pixels
            const int nr_pixels = subgrid_size * subgrid_size;
            int current_nr_pixels = BATCH_SIZE;
            for (int pixel_offset = 0; pixel_offset < nr_pixels; pixel_offset += current_nr_pixels) {
                current_nr_pixels = nr_pixels - pixel_offset < min(nr_threads, BATCH_SIZE) ?
                                    nr_pixels - pixel_offset : min(nr_threads, BATCH_SIZE);

                barrier(CLK_LOCAL_MEM_FENCE);

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
                    pixels_[j] = (float8) (
                        subgrid[idx_xx], subgrid[idx_xy],
                        subgrid[idx_yx], subgrid[idx_yy]);

                    // Compute l,m,n and phase offset
                    const float l = (x+0.5-(subgrid_size/2)) * image_size/subgrid_size;
                    const float m = (y+0.5-(subgrid_size/2)) * image_size/subgrid_size;
                    const float tmp = (l * l) + (m * m);
                    const float n = tmp / (1.0f + native_sqrt(1.0f - tmp));
                    float phase_offset = u_offset*l + v_offset*m + w_offset*n;
                    lmn_phaseoffset_[j] = (float4) (l, m, n, phase_offset);
                } // end for j (pixels)

                barrier(CLK_LOCAL_MEM_FENCE);

                // Iterate current batch of pixels
                for (int k = 0; k < current_nr_pixels; k++) {
                    // Load pixels from local memory
                    float8 pix = pixels_[k];

                    // Compute phase index
                    float4 x = lmn_phaseoffset_[k];
                    float phase_index = dot(uvw_, (float4) (x.s012, 0));

                    // Load phase offset
                    float phase_offset = x.s3;

                    for (int chan = 0; chan < MAX_NR_CHANNELS; chan++) {
                        // Load wavenumber
                        float wavenumber = wavenumbers[channel_offset + chan];

                        // Compute phasor
                        float phase  = (phase_index * wavenumber) - phase_offset;
                        float2 phasor = (float2) (native_cos(phase), native_sin(phase));

                        // Update visibility
                        visibility[chan].even += phasor.x * pix.even;
                        visibility[chan].odd  += phasor.x * pix.odd;
                        visibility[chan].even -= phasor.y * pix.odd;
                        visibility[chan].odd  += phasor.y * pix.even;
                    } // end for chan
                } // end for k (batch)
            } // end for j (pixels)

            for (int chan = 0; chan < MAX_NR_CHANNELS; chan++) {
                // Scale visibility
                visibility[chan] *= (float8) (1.0f / (nr_pixels));

                // Store visibility
                int idx_time = time_offset_global + time;
                int idx_chan = channel_offset + chan;
                int idx_vis = index_visibility(NR_CHANNELS, idx_time, idx_chan);
                __global float8 *vis_ptr = (__global float8 *) &visibilities[idx_vis];
                if (idx_chan < NR_CHANNELS && time < nr_timesteps) {
                    *vis_ptr = visibility[chan];
                }
            } // end for chan
        } // end for time
    } // end for channel_offset
} // end kernel_degridder
