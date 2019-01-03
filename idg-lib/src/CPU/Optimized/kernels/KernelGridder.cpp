#include <cmath>

#if defined(USE_VML)
#define VML_PRECISION VML_LA
#include <mkl_vml.h>
#endif

#include "Types.h"
#include "Math.h"

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
        n_[i] = compute_n(l_[i], m_[i], shift);
    }

    // Iterate all subgrids
    #pragma omp parallel for schedule(guided)
    for (int s = 0; s < nr_subgrids; s++) {
        // Load metadata
        const idg::Metadata m  = metadata[s];
        const int offset       = (m.baseline_offset - baseline_offset_1) + m.time_offset;
        const int nr_timesteps = m.nr_timesteps;
        const int aterm_index  = m.aterm_index;
        const int station1     = m.baseline.station1;
        const int station2     = m.baseline.station2;
        const int x_coordinate = m.coordinate.x;
        const int y_coordinate = m.coordinate.y;
        const float w_offset_in_lambda = w_step_in_lambda * (m.coordinate.z + 0.5);

        // Compute u and v offset in wavelenghts
        const float u_offset = (x_coordinate + subgrid_size/2 - grid_size/2) * (2*M_PI / image_size);
        const float v_offset = (y_coordinate + subgrid_size/2 - grid_size/2) * (2*M_PI / image_size);
        const float w_offset = 2*M_PI * w_offset_in_lambda;

        // Preload visibilities
        const int nr_visibilities = nr_timesteps * nr_channels;
        float vis_xx_real[nr_visibilities] __attribute__((aligned((ALIGNMENT))));
        float vis_xy_real[nr_visibilities] __attribute__((aligned((ALIGNMENT))));
        float vis_yx_real[nr_visibilities] __attribute__((aligned((ALIGNMENT))));
        float vis_yy_real[nr_visibilities] __attribute__((aligned((ALIGNMENT))));
        float vis_xx_imag[nr_visibilities] __attribute__((aligned((ALIGNMENT))));
        float vis_xy_imag[nr_visibilities] __attribute__((aligned((ALIGNMENT))));
        float vis_yx_imag[nr_visibilities] __attribute__((aligned((ALIGNMENT))));
        float vis_yy_imag[nr_visibilities] __attribute__((aligned((ALIGNMENT))));

        for (int vis = 0; vis < nr_visibilities; vis++) {
            int time = vis / nr_channels;
            int chan = vis % nr_channels;
            int time_idx = offset + time;
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

        // Preload uvw
        float uvw_u[nr_timesteps];
        float uvw_v[nr_timesteps];
        float uvw_w[nr_timesteps];

        for (int time = 0; time < nr_timesteps; time++) {
            uvw_u[time] = uvw[offset + time].u;
            uvw_v[time] = uvw[offset + time].v;
            uvw_w[time] = uvw[offset + time].w;
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
            float phase[nr_timesteps*nr_channels];

            for (int time = 0; time < nr_timesteps; time++) {
                // Load UVW coordinates
                float u = uvw_u[time];
                float v = uvw_v[time];
                float w = uvw_w[time];

                // Compute phase index
                float phase_index = u*l_[i] + v*m_[i] + w*n_[i];

                #if defined(__INTEL_COMPILER)
                #pragma vector aligned
                #endif
                for (int chan = 0; chan < nr_channels; chan++) {
                    // Compute phase
                    float wavenumber = wavenumbers[chan];
                    phase[time * nr_channels + chan] = phase_offset[i] - (phase_index * wavenumber);
                }
            } // end time

            // Compute phasor
            float phasor_real[nr_visibilities] __attribute__((aligned((ALIGNMENT))));;
            float phasor_imag[nr_visibilities] __attribute__((aligned((ALIGNMENT))));;
            #if defined(USE_LOOKUP)
            compute_sincos(nr_visibilities, phase, lookup, phasor_imag, phasor_real);
            #else
            compute_sincos(nr_visibilities, phase, phasor_imag, phasor_real);
            #endif

            // Compute pixels
            idg::float2 pixels[NR_POLARIZATIONS];
            compute_reduction(
                nr_visibilities,
                vis_xx_real, vis_xy_real, vis_yx_real, vis_yy_real,
                vis_xx_imag, vis_xy_imag, vis_yx_imag, vis_yy_imag,
                phasor_real, phasor_imag, pixels);

            // Load a term for station1
            size_t station1_idx = index_aterm(subgrid_size, NR_POLARIZATIONS, nr_stations, aterm_index, station1, y, x);
            idg::float2 aXX1 = aterms[station1_idx + 0];
            idg::float2 aXY1 = aterms[station1_idx + 1];
            idg::float2 aYX1 = aterms[station1_idx + 2];
            idg::float2 aYY1 = aterms[station1_idx + 3];

            // Load aterm for station2
            size_t station2_idx = index_aterm(subgrid_size, NR_POLARIZATIONS, nr_stations, aterm_index, station2, y, x);
            idg::float2 aXX2 = aterms[station2_idx + 0];
            idg::float2 aXY2 = aterms[station2_idx + 1];
            idg::float2 aYX2 = aterms[station2_idx + 2];
            idg::float2 aYY2 = aterms[station2_idx + 3];

            // Apply the conjugate transpose of the A-term
            apply_aterm(
                conj(aXX1), conj(aYX1), conj(aXY1), conj(aYY1),
                conj(aXX2), conj(aYX2), conj(aXY2), conj(aYY2),
                pixels);

            if (avg_aterm_correction) apply_avg_aterm_correction(avg_aterm_correction + (y*subgrid_size + x)*16, pixels);

            // Load spheroidal
            float sph = spheroidal[y * subgrid_size + x];

            // Compute shifted position in subgrid
            int x_dst = (x + (subgrid_size/2)) % subgrid_size;
            int y_dst = (y + (subgrid_size/2)) % subgrid_size;

            // Set subgrid value
            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                size_t dst_idx = index_subgrid(NR_POLARIZATIONS, subgrid_size, s, pol, y_dst, x_dst);
                subgrid[dst_idx] = pixels[pol] * sph;
            }
        } // end for i (pixels)
    } // end s
} // end kernel_gridder

} // end extern "C"
