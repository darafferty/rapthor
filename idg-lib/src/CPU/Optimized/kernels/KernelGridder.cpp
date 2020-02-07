#include <cmath>

#if defined(USE_VML)
#define VML_PRECISION VML_LA
#include <mkl_vml.h>
#endif

#include "Types.h"
#include "Index.h"
#include "Math.h"
#include "Memory.h"

inline void update_subgrid(
    int nr_pixels,
    int nr_stations,
    int subgrid_size,
    int subgrid,
    int aterm_index,
    int station1,
    int station2,
    const float*       spheroidal,
    const idg::float2* aterms,
    const idg::float2* avg_aterm_correction,
    const idg::float2* subgrid_local,
          idg::float2* subgrid_global)
{
    // Iterate all pixels in subgrid
    for (int i = 0; i < nr_pixels; i++) {
        int y = i / subgrid_size;
        int x = i % subgrid_size;

        // Apply the conjugate transpose of the A-term
        size_t station1_idx = index_aterm(subgrid_size, nr_stations, aterm_index, station1, y, x, 0);
        size_t station2_idx = index_aterm(subgrid_size, nr_stations, aterm_index, station2, y, x, 0);
        idg::float2 *aterm1_ptr = (idg::float2 *) &aterms[station1_idx];
        idg::float2 *aterm2_ptr = (idg::float2 *) &aterms[station2_idx];
        idg::float2 pixels[NR_POLARIZATIONS];
        for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
            pixels[pol] = subgrid_local[pol * nr_pixels + i];
        }
        apply_aterm_gridder(pixels, aterm1_ptr, aterm2_ptr);

        if (avg_aterm_correction) apply_avg_aterm_correction(avg_aterm_correction + (y*subgrid_size + x)*16, pixels);

        // Compute shifted position in subgrid
        int x_dst = (x + (subgrid_size/2)) % subgrid_size;
        int y_dst = (y + (subgrid_size/2)) % subgrid_size;

        // Load spheroidal
        float sph = spheroidal[y * subgrid_size + x];

        // Update global subgrid
        for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
            size_t dst_idx = index_subgrid(subgrid_size, subgrid, pol, y_dst, x_dst);
            subgrid_global[dst_idx] += pixels[pol] * sph;
        }
    }
}

extern "C" {

void kernel_gridder(
    const int                        nr_subgrids,
    const int                        grid_size,
    const int                        subgrid_size,
    const float                      image_size,
    const float                      w_step_in_lambda,
    const float* __restrict__        shift,
    const int                        nr_channels,
    const int                        nr_stations,
    const idg::UVW<float>*           uvw,
    const float*                     wavenumbers,
    const idg::float2*               visibilities,
    const float*                     spheroidal,
    const idg::float2*               aterms,
    const int*                       aterms_indices,
    const idg::float2*               avg_aterm_correction,
    const idg::Metadata*             metadata,
          idg::float2*               subgrid)
{
    #if defined(USE_LOOKUP)
    initialize_lookup();
    #endif

    // Compute l,m,n
    const unsigned nr_pixels = subgrid_size*subgrid_size;
    float l_[nr_pixels];
    float m_[nr_pixels];
    float n_[nr_pixels];

    for (unsigned i = 0; i < nr_pixels; i++) {
        int y = i / subgrid_size;
        int x = i % subgrid_size;

        l_[i] = compute_l(x, subgrid_size, image_size);
        m_[i] = compute_m(y, subgrid_size, image_size);
        n_[i] = compute_n(-l_[i], m_[i], shift);
    }

    // Iterate all subgrids
    #pragma omp parallel for schedule(guided)
    for (int s = 0; s < nr_subgrids; s++) {
        // Initialize global subgrid
        size_t subgrid_idx = index_subgrid(subgrid_size, s, 0, 0, 0);
        idg::float2 *subgrid_ptr = &subgrid[subgrid_idx];
        memset(subgrid_ptr, 0, NR_POLARIZATIONS*nr_pixels*sizeof(idg::float2));

        // Load metadata
        const idg::Metadata m  = metadata[s];
        const int time_offset_global = m.time_index;
        const int nr_timesteps  = m.nr_timesteps;
        const int channel_begin = m.channel_begin;
        const int channel_end   = m.channel_end;
        const int nr_channels_subgrid = channel_end - channel_begin;
        const int station1      = m.baseline.station1;
        const int station2      = m.baseline.station2;
        const int x_coordinate  = m.coordinate.x;
        const int y_coordinate  = m.coordinate.y;
        const float w_offset_in_lambda = w_step_in_lambda * (m.coordinate.z + 0.5);

        // Allocate memory
        size_t total_nr_visibilities = nr_timesteps * nr_channels_subgrid;
        float* vis_xx_real = allocate_memory<float>(total_nr_visibilities);
        float* vis_xy_real = allocate_memory<float>(total_nr_visibilities);
        float* vis_yx_real = allocate_memory<float>(total_nr_visibilities);
        float* vis_yy_real = allocate_memory<float>(total_nr_visibilities);
        float* vis_xx_imag = allocate_memory<float>(total_nr_visibilities);
        float* vis_xy_imag = allocate_memory<float>(total_nr_visibilities);
        float* vis_yx_imag = allocate_memory<float>(total_nr_visibilities);
        float* vis_yy_imag = allocate_memory<float>(total_nr_visibilities);
        float* phasor_real = allocate_memory<float>(total_nr_visibilities);
        float* phasor_imag = allocate_memory<float>(total_nr_visibilities);
        float* phase       = allocate_memory<float>(total_nr_visibilities);
        float* phase_offset = allocate_memory<float>(nr_pixels);
        float* uvw_u = allocate_memory<float>(nr_timesteps);
        float* uvw_v = allocate_memory<float>(nr_timesteps);
        float* uvw_w = allocate_memory<float>(nr_timesteps);
        idg::float2* subgrid_local = allocate_memory<idg::float2>(NR_POLARIZATIONS * nr_pixels);

        // Initialize local subgrid
        memset(subgrid_local, 0, NR_POLARIZATIONS*nr_pixels*sizeof(idg::float2));

        // Initialize aterm index to first timestep
        int aterm_idx_previous = aterms_indices[time_offset_global];

        // Compute u and v offset in wavelenghts
        const float u_offset = (x_coordinate + subgrid_size/2 - grid_size/2) * (2*M_PI / image_size);
        const float v_offset = (y_coordinate + subgrid_size/2 - grid_size/2) * (2*M_PI / image_size);
        const float w_offset = 2*M_PI * w_offset_in_lambda;

        // Iterate all timesteps
        int current_nr_timesteps = 0;
        for (int time_offset_local = 0; time_offset_local < nr_timesteps; time_offset_local += current_nr_timesteps) {

            // Get aterm indices for current timestep
            int time_current = time_offset_global + time_offset_local;
            int aterm_idx_current = aterms_indices[time_current];

            // Determine whether aterm has changed
            #if defined(__PPC__) // workaround compiler bug
            unsigned int aterm_changed;
            #else
            bool aterm_changed;
            #endif
            aterm_changed = aterm_idx_previous != aterm_idx_current;

            // Determine number of timesteps to process
            current_nr_timesteps = 0;
            for (int time = time_offset_local; time < nr_timesteps; time++) {
                if (aterms_indices[time_offset_global + time] == aterm_idx_current) {
                    current_nr_timesteps++;
                } else {
                    break;
                }
            }

            if (aterm_changed) {
                // Update subgrid
                update_subgrid(
                    nr_pixels, nr_stations, subgrid_size, s,
                    aterm_idx_previous, station1, station2,
                    spheroidal, aterms, avg_aterm_correction,
                    (const idg::float2*) subgrid_local, subgrid);

                // Reset local subgrid for new aterms
                memset(subgrid_local, 0, NR_POLARIZATIONS*nr_pixels*sizeof(idg::float2));

                // Update aterm indices
                aterm_idx_previous = aterm_idx_current;
            }

            // Load visibilities
            for (int time = 0; time < current_nr_timesteps; time++) {
                for (int chan = channel_begin; chan < channel_end; chan++) {
                    int time_idx = time_offset_global + time_offset_local + time;
                    int chan_idx = chan - channel_begin;
                    size_t src_idx = index_visibility(nr_channels, time_idx, chan, 0);
                    size_t dst_idx = time * nr_channels_subgrid + chan_idx;

                    vis_xx_real[dst_idx] = visibilities[src_idx + 0].real;
                    vis_xx_imag[dst_idx] = visibilities[src_idx + 0].imag;
                    vis_xy_real[dst_idx] = visibilities[src_idx + 1].real;
                    vis_xy_imag[dst_idx] = visibilities[src_idx + 1].imag;
                    vis_yx_real[dst_idx] = visibilities[src_idx + 2].real;
                    vis_yx_imag[dst_idx] = visibilities[src_idx + 2].imag;
                    vis_yy_real[dst_idx] = visibilities[src_idx + 3].real;
                    vis_yy_imag[dst_idx] = visibilities[src_idx + 3].imag;
                }
            }

            // Preload uvw
            for (int time = 0; time < current_nr_timesteps; time++) {
                int time_idx = time_offset_global + time_offset_local + time;
                uvw_u[time] = uvw[time_idx].u;
                uvw_v[time] = uvw[time_idx].v;
                uvw_w[time] = uvw[time_idx].w;
            }

            // Compute phase offset
            for (unsigned i = 0; i < nr_pixels; i++) {
                phase_offset[i] = u_offset*l_[i] + v_offset*m_[i] + w_offset*n_[i];
            }

            // Iterate all pixels in subgrid
            for (unsigned i = 0; i < nr_pixels; i++) {
                int y = i / subgrid_size;
                int x = i % subgrid_size;

                // Compute phase
                float phase[current_nr_timesteps*nr_channels_subgrid] __attribute__((aligned(ALIGNMENT)));

                for (int time = 0; time < current_nr_timesteps; time++) {
                    // Load UVW coordinates
                    float u = uvw_u[time];
                    float v = uvw_v[time];
                    float w = uvw_w[time];

                    // Compute phase index
                    float phase_index = u*l_[i] + v*m_[i] + w*n_[i];

                    // pragma vector aligned
                    for (int chan = channel_begin; chan < channel_end; chan++) {
                        int chan_idx = chan - channel_begin;
                        // Compute phase
                        float wavenumber = wavenumbers[chan];
                        phase[time * nr_channels_subgrid + chan_idx] = phase_offset[i] - (phase_index * wavenumber);
                    }
                } // end time

                size_t current_nr_visibilities = current_nr_timesteps * nr_channels_subgrid;

                // Compute phasor
                compute_sincos(current_nr_visibilities, phase, phasor_imag, phasor_real);

                // Compute pixels
                idg::float2 pixels[NR_POLARIZATIONS] __attribute__((aligned(ALIGNMENT)));
                compute_reduction(
                    current_nr_visibilities,
                    vis_xx_real, vis_xy_real, vis_yx_real, vis_yy_real,
                    vis_xx_imag, vis_xy_imag, vis_yx_imag, vis_yy_imag,
                    phasor_real, phasor_imag, pixels);

                // Update local subgrid
                for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                    size_t idx = index_subgrid(subgrid_size, 0, pol, y, x);
                    subgrid_local[idx] += pixels[pol];
                }
            } // end for i (pixels)
        } // end time_offset_local

        update_subgrid(
            nr_pixels, nr_stations, subgrid_size, s,
            aterm_idx_previous, station1, station2,
            spheroidal, aterms, avg_aterm_correction,
            (const idg::float2*) subgrid_local, subgrid);

        // Free memory
        free(vis_xx_real);
        free(vis_xy_real);
        free(vis_yx_real);
        free(vis_yy_real);
        free(vis_xx_imag);
        free(vis_xy_imag);
        free(vis_yx_imag);
        free(vis_yy_imag);
        free(phase);
        free(phase_offset);
        free(phasor_real);
        free(phasor_imag);
        free(uvw_u);
        free(uvw_v);
        free(uvw_w);
        free(subgrid_local);
    } // end s
} // end kernel_gridder

} // end extern "C"
