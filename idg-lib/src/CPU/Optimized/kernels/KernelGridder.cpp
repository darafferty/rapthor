#include <cmath>

#if defined(USE_VML)
#define VML_PRECISION VML_LA
#include <mkl_vml.h>
#endif

#include "Types.h"
#include "Math.h"

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
        size_t station1_idx = index_aterm(subgrid_size, NR_POLARIZATIONS, nr_stations, aterm_index, station1, y, x);
        size_t station2_idx = index_aterm(subgrid_size, NR_POLARIZATIONS, nr_stations, aterm_index, station2, y, x);
        idg::float2 *aterm1_ptr = (idg::float2 *) &aterms[station1_idx];
        idg::float2 *aterm2_ptr = (idg::float2 *) &aterms[station2_idx];
        idg::float2 pixels[NR_POLARIZATIONS];
        for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
            pixels[pol] = subgrid_local[pol * nr_pixels + i];
        }
        #if 1
        apply_aterm_gridder(pixels, aterm1_ptr, aterm2_ptr);
        #else
        idg::float2 aterm1[4];
        idg::float2 aterm2[4];
        conjugate(aterm1_ptr, aterm1);
        hermitian(aterm2_ptr, aterm2);
        apply_aterm_generic(pixels, aterm1, aterm2);
        #endif

        if (avg_aterm_correction) apply_avg_aterm_correction(avg_aterm_correction + (y*subgrid_size + x)*16, pixels);

        // Compute shifted position in subgrid
        int x_dst = (x + (subgrid_size/2)) % subgrid_size;
        int y_dst = (y + (subgrid_size/2)) % subgrid_size;

        // Load spheroidal
        float sph = spheroidal[y * subgrid_size + x];

        // Update global subgrid
        for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
            size_t dst_idx = index_subgrid(NR_POLARIZATIONS, subgrid_size, subgrid, pol, y_dst, x_dst);
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
    const idg::UVWCoordinate<float>* uvw,
    const float*                     wavenumbers,
    const idg::float2*               visibilities,
    const float*                     spheroidal,
    const idg::float2*               aterms,
    const idg::float2*               avg_aterm_correction,
    const idg::Metadata*             metadata,
          idg::float2*               subgrid)
{
    #if defined(USE_LOOKUP)
    CREATE_LOOKUP
    #endif

    // Find offset of first subgrid
    const idg::Metadata m       = metadata[0];
    const int baseline_offset_1 = m.baseline_offset;

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
        size_t subgrid_idx = index_subgrid(NR_POLARIZATIONS, subgrid_size, s, 0, 0, 0);
        idg::float2 *subgrid_ptr = &subgrid[subgrid_idx];
        memset(subgrid_ptr, 0, NR_POLARIZATIONS*nr_pixels*sizeof(idg::float2));

        // Initialize local subgrid
        idg::float2 subgrid_local[NR_POLARIZATIONS][subgrid_size][subgrid_size];
        memset(subgrid_local, 0, NR_POLARIZATIONS*nr_pixels*sizeof(idg::float2));

        // Load metadata
        const idg::Metadata m  = metadata[s];
        const int time_offset_global = (m.baseline_offset - baseline_offset_1) + m.time_offset;
        const int nr_timesteps = m.nr_timesteps;
        const int aterm_index  = m.aterm_index;
        const int station1     = m.baseline.station1;
        const int station2     = m.baseline.station2;
        const int x_coordinate = m.coordinate.x;
        const int y_coordinate = m.coordinate.y;
        const float w_offset_in_lambda = w_step_in_lambda * (m.coordinate.z + 0.5);

        // Initialize aterm indices to first timestep
        size_t aterm1_idx_previous = 0;
        size_t aterm2_idx_previous = 0;

        // Compute u and v offset in wavelenghts
        const float u_offset = (x_coordinate + subgrid_size/2 - grid_size/2) * (2*M_PI / image_size);
        const float v_offset = (y_coordinate + subgrid_size/2 - grid_size/2) * (2*M_PI / image_size);
        const float w_offset = 2*M_PI * w_offset_in_lambda;

        // Iterate all timesteps
        int current_nr_timesteps = 1;
        for (int time_offset_local = 0; time_offset_local < nr_timesteps; time_offset_local += current_nr_timesteps) {
            // Get aterm indices for current timestep
            size_t aterm1_idx_current = 0;
            size_t aterm2_idx_current = 0;

            // Determine whether aterm has changed
            bool aterm_changed = aterm1_idx_previous != aterm1_idx_current ||
                                 aterm2_idx_previous != aterm2_idx_current;

            // Determine number of timesteps to process
            current_nr_timesteps = nr_timesteps - time_offset_local; // TODO
            int current_nr_visibilities = current_nr_timesteps * nr_channels;

            if (aterm_changed) {
                // Update subgrid
                update_subgrid(
                    nr_pixels, nr_stations, subgrid_size, s,
                    aterm_index, station1, station2,
                    spheroidal, aterms, avg_aterm_correction,
                    (const idg::float2*) subgrid_local, subgrid);

                // Reset local subgrid for new aterms
                memset(subgrid_local, 0, NR_POLARIZATIONS*nr_pixels*sizeof(idg::float2));

                // Update aterm indices
                aterm1_idx_previous = aterm1_idx_current;
                aterm2_idx_previous = aterm2_idx_current;
            }

            // Load visibilities
            float vis_xx_real[current_nr_visibilities] __attribute__((aligned((ALIGNMENT))));
            float vis_xy_real[current_nr_visibilities] __attribute__((aligned((ALIGNMENT))));
            float vis_yx_real[current_nr_visibilities] __attribute__((aligned((ALIGNMENT))));
            float vis_yy_real[current_nr_visibilities] __attribute__((aligned((ALIGNMENT))));
            float vis_xx_imag[current_nr_visibilities] __attribute__((aligned((ALIGNMENT))));
            float vis_xy_imag[current_nr_visibilities] __attribute__((aligned((ALIGNMENT))));
            float vis_yx_imag[current_nr_visibilities] __attribute__((aligned((ALIGNMENT))));
            float vis_yy_imag[current_nr_visibilities] __attribute__((aligned((ALIGNMENT))));

            for (int time = 0; time < current_nr_timesteps; time++) {
                for (int chan = 0; chan < nr_channels; chan++) {
                    int time_idx = time_offset_global + time_offset_local + time;
                    int chan_idx = chan;
                    size_t src_idx = index_visibility(nr_channels, NR_POLARIZATIONS, time_idx, chan_idx, 0);
                    size_t dst_idx = time * nr_channels + chan;

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
            float uvw_u[current_nr_timesteps];
            float uvw_v[current_nr_timesteps];
            float uvw_w[current_nr_timesteps];

            for (int time = 0; time < current_nr_timesteps; time++) {
                int time_idx = time_offset_global + time_offset_local + time;
                uvw_u[time] = uvw[time_idx].u;
                uvw_v[time] = uvw[time_idx].v;
                uvw_w[time] = uvw[time_idx].w;
            }

            // Compute phase offset
            float phase_offset[nr_pixels];

            for (unsigned i = 0; i < nr_pixels; i++) {
                phase_offset[i] = u_offset*l_[i] + v_offset*m_[i] + w_offset*n_[i];
            }

            // Iterate all pixels in subgrid
            for (unsigned i = 0; i < nr_pixels; i++) {
                int y = i / subgrid_size;
                int x = i % subgrid_size;

                // Compute phase
                float phase[current_nr_timesteps*nr_channels];

                for (int time = 0; time < current_nr_timesteps; time++) {
                    // Load UVW coordinates
                    float u = uvw_u[time];
                    float v = uvw_v[time];
                    float w = uvw_w[time];

                    // Compute phase index
                    float phase_index = u*l_[i] + v*m_[i] + w*n_[i];

                    // pragma vector aligned
                    for (int chan = 0; chan < nr_channels; chan++) {
                        // Compute phase
                        float wavenumber = wavenumbers[chan];
                        phase[time * nr_channels + chan] = phase_offset[i] - (phase_index * wavenumber);
                    }
                } // end time

                // Compute phasor
                float phasor_real[current_nr_visibilities] __attribute__((aligned(ALIGNMENT)));
                float phasor_imag[current_nr_visibilities] __attribute__((aligned(ALIGNMENT)));
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
                    subgrid_local[pol][y][x] += pixels[pol];
                }
            } // end for i (pixels)
        } // end time_offset_local

        update_subgrid(
            nr_pixels, nr_stations, subgrid_size, s,
            aterm_index, station1, station2,
            spheroidal, aterms, avg_aterm_correction,
            (const idg::float2*) subgrid_local, subgrid);

    } // end s
} // end kernel_gridder

} // end extern "C"
