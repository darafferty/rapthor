#include <cmath>

#if defined(USE_VML)
#define VML_PRECISION VML_LA
#include <mkl_vml.h>
#endif

#include "Types.h"
#include "Math.h"

template<int current_nr_channels>
void kernel_degridder_(
    const int                        nr_subgrids,
    const int                        grid_size,
    const int                        subgrid_size,
    const float                      image_size,
    const float                      w_step_in_lambda,
    const int                        nr_channels,
    const int                        channel_offset,
    const int                        nr_stations,
    const idg::UVWCoordinate<float>* uvw,
    const float*                     wavenumbers,
          idg::float2*               visibilities,
    const float*                     spheroidal,
    const idg::float2*               aterms,
    const idg::Metadata*             metadata,
    const idg::float2*               subgrid)
{
    #if defined(USE_LOOKUP)
    float lookup[NR_SAMPLES];
    compute_lookup(lookup);
    #endif

    // Find offset of first subgrid
    const idg::Metadata m       = metadata[0];
    const int baseline_offset_1 = m.baseline_offset;

    // Iterate all subgrids
    #pragma omp parallel for
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

        // Storage
        const int nr_pixels = subgrid_size*subgrid_size;
        float pixels_xx_real[nr_pixels] __attribute__((aligned((ALIGNMENT))));
        float pixels_xy_real[nr_pixels] __attribute__((aligned((ALIGNMENT))));
        float pixels_yx_real[nr_pixels] __attribute__((aligned((ALIGNMENT))));
        float pixels_yy_real[nr_pixels] __attribute__((aligned((ALIGNMENT))));
        float pixels_xx_imag[nr_pixels] __attribute__((aligned((ALIGNMENT))));
        float pixels_xy_imag[nr_pixels] __attribute__((aligned((ALIGNMENT))));
        float pixels_yx_imag[nr_pixels] __attribute__((aligned((ALIGNMENT))));
        float pixels_yy_imag[nr_pixels] __attribute__((aligned((ALIGNMENT))));

        // Apply aterm to subgrid
        for (int i = 0; i < nr_pixels; i++) {
            int y = i / subgrid_size;
            int x = i % subgrid_size;

            // Load aterm for station1
            int station1_idx = index_aterm(subgrid_size, NR_POLARIZATIONS, nr_stations, aterm_index, station1, y, x);
            idg::float2 aXX1 = aterms[station1_idx + 0];
            idg::float2 aXY1 = aterms[station1_idx + 1];
            idg::float2 aYX1 = aterms[station1_idx + 2];
            idg::float2 aYY1 = aterms[station1_idx + 3];

            // Load aterm for station2
            int station2_idx = index_aterm(subgrid_size, NR_POLARIZATIONS, nr_stations, aterm_index, station2, y, x);
            idg::float2 aXX2 = conj(aterms[station2_idx + 0]);
            idg::float2 aXY2 = conj(aterms[station2_idx + 1]);
            idg::float2 aYX2 = conj(aterms[station2_idx + 2]);
            idg::float2 aYY2 = conj(aterms[station2_idx + 3]);

            // Load spheroidal
            float _spheroidal = spheroidal[y * subgrid_size + x];

            // Compute shifted position in subgrid
            int x_src = (x + (subgrid_size/2)) % subgrid_size;
            int y_src = (y + (subgrid_size/2)) % subgrid_size;

            // Load pixel values and apply spheroidal
            idg::float2 pixels[NR_POLARIZATIONS];
            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                int src_idx = index_subgrid(NR_POLARIZATIONS, subgrid_size, s, pol, y_src, x_src);
                pixels[pol] = _spheroidal * subgrid[src_idx];
            }

            apply_aterm(
                aXX1, aXY1, aYX1, aYY1,
                aXX2, aXY2, aYX2, aYY2,
                pixels);

            // Store pixels
            pixels_xx_real[i] = pixels[0].real;
            pixels_xy_real[i] = pixels[1].real;
            pixels_yx_real[i] = pixels[2].real;
            pixels_yy_real[i] = pixels[3].real;
            pixels_xx_imag[i] = pixels[0].imag;
            pixels_xy_imag[i] = pixels[1].imag;
            pixels_yx_imag[i] = pixels[2].imag;
            pixels_yy_imag[i] = pixels[3].imag;
        }

        // Compute u and v offset in wavelenghts
        const float u_offset = (x_coordinate + subgrid_size/2 - grid_size/2)
                               * (2*M_PI / image_size);
        const float v_offset = (y_coordinate + subgrid_size/2 - grid_size/2)
                               * (2*M_PI / image_size);
        const float w_offset = 2*M_PI * w_offset_in_lambda;

        // Iterate all timesteps
        for (int time = 0; time < nr_timesteps; time++) {
            // Load UVW coordinates
            float u = uvw[offset + time].u;
            float v = uvw[offset + time].v;
            float w = uvw[offset + time].w;

            float phase_index[nr_pixels];
            float phase_offset[nr_pixels];

            for (int i = 0; i < nr_pixels; i++) {
                int y = i / subgrid_size;
                int x = i % subgrid_size;

                // Compute l,m,n
                const float l = (x+0.5-(subgrid_size/2)) * image_size/subgrid_size;
                const float m = (y+0.5-(subgrid_size/2)) * image_size/subgrid_size;
                // evaluate n = 1.0f - sqrt(1.0 - (l * l) - (m * m));
                // accurately for small values of l and m
                const float tmp = (l * l) + (m * m);
                const float n = tmp / (1.0f + sqrtf(1.0f - tmp));

                // Compute phase index
                phase_index[i] = u*l + v*m + w*n;

                // Compute phase offset
                phase_offset[i] = u_offset*l + v_offset*m + w_offset*n;
            }

            // Iterate all channels
            for (int chan = 0; chan < current_nr_channels; chan++) {
                // Compute phase
                float phase[nr_pixels];

                for (int i = 0; i < nr_pixels; i++) {
                    // Compute phase
                    float wavenumber = wavenumbers[channel_offset + chan];
                    phase[i] = (phase_index[i] * wavenumber) - phase_offset[i];
                }

                // Compute phasor
                float phasor_real[nr_pixels] __attribute__((aligned((ALIGNMENT))));
                float phasor_imag[nr_pixels] __attribute__((aligned((ALIGNMENT))));
                #if defined(USE_LOOKUP)
                compute_sincos(nr_pixels, phase, lookup, phasor_imag, phasor_real);
                #else
                compute_sincos(nr_pixels, phase, phasor_imag, phasor_real);
                #endif

                // Compute visibilities
                idg::float2 sums[NR_POLARIZATIONS];

                compute_reduction(
                    nr_pixels,
                    pixels_xx_real, pixels_xy_real, pixels_yx_real, pixels_yy_real,
                    pixels_xx_imag, pixels_xy_imag, pixels_yx_imag, pixels_yy_imag,
                    phasor_real, phasor_imag, sums);

                // Store visibilities
                const float scale = 1.0f / nr_pixels;
                int time_idx = offset + time;
                int chan_idx = channel_offset + chan;
                int dst_idx = index_visibility( nr_channels, NR_POLARIZATIONS, time_idx, chan_idx, 0);
                for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                    visibilities[dst_idx+pol] = {scale*sums[pol].real, scale*sums[pol].imag};
                }
            } // end for channel
        } // end for time
    } // end #pragma parallel
} // end kernel_degridder_

extern "C" {

void kernel_degridder(
    const int                        nr_subgrids,
    const int                        grid_size,
    const int                        subgrid_size,
    const float                      image_size,
    const float                      w_offset_in_lambda,
    const int                        nr_channels,
    const int                        nr_stations,
    const idg::UVWCoordinate<float>* uvw,
    const float*                     wavenumbers,
          idg::float2*               visibilities,
    const float*                     spheroidal,
    const idg::float2*               aterms,
    const idg::Metadata*             metadata,
    const idg::float2*               subgrid)
{
    int channel_offset = 0;

    for (; (channel_offset + 16) <= nr_channels; channel_offset += 16) {
        kernel_degridder_<16>(
            nr_subgrids, grid_size, subgrid_size, image_size, w_offset_in_lambda,
            nr_channels, channel_offset, nr_stations,
            uvw, wavenumbers, visibilities, spheroidal, aterms, metadata, subgrid);
    }

    for (; (channel_offset + 8) <= nr_channels; channel_offset += 8) {
        kernel_degridder_<8>(
            nr_subgrids, grid_size, subgrid_size, image_size, w_offset_in_lambda,
            nr_channels, channel_offset, nr_stations,
            uvw, wavenumbers, visibilities, spheroidal, aterms, metadata, subgrid);
    }

    for (; (channel_offset + 4) <= nr_channels; channel_offset += 4) {
        kernel_degridder_<4>(
            nr_subgrids, grid_size, subgrid_size, image_size, w_offset_in_lambda,
            nr_channels, channel_offset, nr_stations,
            uvw, wavenumbers, visibilities, spheroidal, aterms, metadata, subgrid);
    }

    for (; (channel_offset + 1) <= nr_channels; channel_offset += 1) {
        kernel_degridder_<1>(
            nr_subgrids, grid_size, subgrid_size, image_size, w_offset_in_lambda,
            nr_channels, channel_offset, nr_stations,
            uvw, wavenumbers, visibilities, spheroidal, aterms, metadata, subgrid);
    }
}

} // end extern "C"
