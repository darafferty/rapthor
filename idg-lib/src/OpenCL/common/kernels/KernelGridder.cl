#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))

#define UNROLL_PIXELS   4

/*
    Kernel
*/
__kernel
__attribute__((work_group_size_hint(BLOCK_SIZE, 1, 1)))
void kernel_gridder(
    const int                grid_size,
    const int                subgrid_size,
    const float              image_size,
    const float              w_step,
    const int                nr_stations,
    __global const UVW*      uvw,
    __global const float*    wavenumbers,
    __global const float2*   visibilities,
    __global const float*    spheroidal,
    __global const float2*   aterm,
    __global const Metadata* metadata,
    __global       float2*   subgrid)
{
    int tidx = get_local_id(0);
    int tidy = get_local_id(1);
    int tid = tidx + tidy * get_local_size(0);;
    int nr_threads = get_local_size(0) * get_local_size(1);
    int s = get_group_id(0);

    // Local memory
    __local float8 visibilities_[BATCH_SIZE][NR_CHANNELS];
    __local float4 uvw_[BATCH_SIZE/NR_CHANNELS];
    __local float  wavenumbers_[NR_CHANNELS];

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
    float4 uvw_offset = (float4) (
        (x_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI,
        (y_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI,
        w_step * ((float)m.coordinate.z + 0.5) * 2 * M_PI,
        0);

    // Load wavenumbers
    for (int i = tid; i < NR_CHANNELS; i += nr_threads) {
        wavenumbers_[i] = wavenumbers[i];
    }

    // Iterate all pixels in subgrid
    for (int i = tid; i < ALIGN(subgrid_size * subgrid_size, nr_threads); i += nr_threads * UNROLL_PIXELS) {
        // Private pixels
        float8 pixels[UNROLL_PIXELS];

        for (int j = 0; j < UNROLL_PIXELS; j++) {
            pixels[j] = (float8) 0;
        }

        // Compute l,m,n, phase_offset
        float4 lmn[UNROLL_PIXELS];
        float phase_offset[UNROLL_PIXELS];

        for (int j = 0; j < UNROLL_PIXELS; j++) {
            int i_ = i + j * nr_threads;
            int y = i_ / subgrid_size;
            int x = i_ % subgrid_size;
            float l = (x+0.5-(subgrid_size/2)) * image_size/subgrid_size;
            float m = (y+0.5-(subgrid_size/2)) * image_size/subgrid_size;
            float tmp = (l * l) + (m * m);
            float n = tmp / (1.0f + native_sqrt(1.0f - tmp));
            lmn[j] = (float4) (l, m, n, 0);
            phase_offset[j] = dot(uvw_offset, lmn[j]);
        }

        // Iterate timesteps
        int current_nr_timesteps = BATCH_SIZE;
        for (int time_offset_local = 0; time_offset_local < nr_timesteps; time_offset_local += current_nr_timesteps) {
            current_nr_timesteps = nr_timesteps - time_offset_local < BATCH_SIZE ?
                                   nr_timesteps - time_offset_local : BATCH_SIZE;

            barrier(CLK_LOCAL_MEM_FENCE);

            // Load UVW
            for (int time = tid; time < current_nr_timesteps; time += nr_threads) {
                int idx_time = time_offset_global + time_offset_local + time;
                uvw_[time].x = uvw[idx_time].u;
                uvw_[time].y = uvw[idx_time].v;
                uvw_[time].z = uvw[idx_time].w;
            }

            // Load visibilities
            for (int i = tid; i < current_nr_timesteps * NR_CHANNELS; i += nr_threads) {
                int idx_time = time_offset_global + time_offset_local + (i / NR_CHANNELS);
                int idx_chan = i % NR_CHANNELS;
                int idx_vis = index_visibility(NR_CHANNELS, idx_time, idx_chan);
                float2 a = visibilities[idx_vis + 0];
                float2 b = visibilities[idx_vis + 1];
                float2 c = visibilities[idx_vis + 2];
                float2 d = visibilities[idx_vis + 3];
                visibilities_[0][i] = (float8) (a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y);
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            // Iterate current batch of timesteps
            for (int time = 0; time < current_nr_timesteps; time++) {
                // Load UVW coordinates
                float4 t = uvw_[time];

                // Compute phase index and phase offset
                float phase_index[UNROLL_PIXELS];

                for (int j = 0; j < UNROLL_PIXELS; j++) {
                    phase_index[j]  = dot(t, lmn[j]);
                }

                for (int chan = 0; chan < NR_CHANNELS; chan++) {
                    float wavenumber = wavenumbers_[chan];

                    // Load visibilities from shared memory
                    float8 vis = visibilities_[time][chan];

                    for (int j = 0; j < UNROLL_PIXELS; j++) {
                        // Compute phasor
                        float phase   = phase_offset[j] - (phase_index[j] * wavenumber);
                        float2 phasor = (float2) (native_cos(phase), native_sin(phase));

                        // Update pixel
                        pixels[j].even += phasor.x * vis.even;
                        pixels[j].odd  += phasor.x * vis.odd;
                        pixels[j].even -= phasor.y * vis.odd;
                        pixels[j].odd  += phasor.y * vis.even;
                    }
                } // end for chan
            } // end for time
        } // end for time_offset_local

        for (int j = 0; j < UNROLL_PIXELS; j++) {
            int i_ = i + j * nr_threads;
            if (i_ < subgrid_size * subgrid_size) {
                int y = i_ / subgrid_size;
                int x = i_ % subgrid_size;

                // Get aterm for station1
                int station1_idx = index_aterm(subgrid_size, nr_stations, aterm_index, station1, y, x);
                float2 aXX1 = aterm[station1_idx + 0];
                float2 aXY1 = aterm[station1_idx + 1];
                float2 aYX1 = aterm[station1_idx + 2];
                float2 aYY1 = aterm[station1_idx + 3];
                float8 aterm1 = (float8) (aXX1, aYY1, aXX1, aYY1);
                float8 aterm2 = (float8) (aYX1, aXY1, aYX1, aXY1);

                // Get aterm for station2
                int station2_idx = index_aterm(subgrid_size, nr_stations, aterm_index, station2, y, x);
                float2 aXX2 = conj(aterm[station2_idx + 0]);
                float2 aXY2 = conj(aterm[station2_idx + 1]);
                float2 aYX2 = conj(aterm[station2_idx + 2]);
                float2 aYY2 = conj(aterm[station2_idx + 3]);
                float8 aterm3 = (float8) (aXX2, aXX2, aYY2, aYY2);
                float8 aterm4 = (float8) (aYX2, aYX2, aXY2, aXY2);

                float8 pixels_aterm;

                // Apply aterm to pixels: P*A1
                // [ uvXX, uvXY;    [ aXX1, aXY1;
                //   uvYX, uvYY ] *   aYX1, aYY1 ]
                pixels_aterm = (float8) (0);
                pixels_aterm += cmul8(pixels[j], aterm1);
                pixels_aterm += cmul8(pixels[j], aterm2);

                // Apply aterm to pixels: A2^H*P
                // [ aXX2, aYX1;      [ uvXX, uvXY;
                //   aXY1, aYY2 ]  *    uvYX, uvYY ]
                pixels[j] = pixels_aterm;
                pixels_aterm = (float8) (0);
                pixels_aterm += cmul8(pixels[j], aterm3);
                pixels_aterm += cmul8(pixels[j], aterm4);

                // Load spheroidal
                pixels_aterm *= spheroidal[y * subgrid_size + x];

                // Compute shifted position in subgrid
                int x_dst = (x + (subgrid_size/2)) % subgrid_size;
                int y_dst = (y + (subgrid_size/2)) % subgrid_size;

                // Set subgrid value
                int idx_xx = index_subgrid(subgrid_size, s, 0, y_dst, x_dst);
                int idx_xy = index_subgrid(subgrid_size, s, 1, y_dst, x_dst);
                int idx_yx = index_subgrid(subgrid_size, s, 2, y_dst, x_dst);
                int idx_yy = index_subgrid(subgrid_size, s, 3, y_dst, x_dst);
                subgrid[idx_xx] = pixels_aterm.s01;
                subgrid[idx_xy] = pixels_aterm.s23;
                subgrid[idx_yx] = pixels_aterm.s45;
                subgrid[idx_yy] = pixels_aterm.s67;
            }
        }
    } // end for i (pixels)
} // end kernel_gridder
