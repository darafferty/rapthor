#include <cmath>

#if defined(USE_VML)
#define VML_PRECISION VML_LA
#include <mkl_vml.h>
#endif

#include "Types.h"
#include "Math.h"

template<int current_nr_channels>
void kernel_gridder_(
    const int                       nr_subgrids,
    const int                       grid_size,
    const int                       subgrid_size,
    const float                     image_size,
    const float                     w_step_in_lambda,
    const int                       nr_channels,
    const int                       channel_offset,
    const int                       nr_stations,
    const idg::UVWCoordinate<float> uvw[],
    const float                     wavenumbers[],
    const idg::float2               visibilities[][NR_POLARIZATIONS],
    const float                     spheroidal[subgrid_size][subgrid_size],
    const idg::float2               aterms[][subgrid_size][subgrid_size][NR_POLARIZATIONS],
    const idg::Metadata             metadata[],
          idg::float2               subgrid[][NR_POLARIZATIONS][subgrid_size][subgrid_size]
    )
{
    // Find offset of first subgrid
    const idg::Metadata m       = metadata[0];
    const int baseline_offset_1 = m.baseline_offset;
    const int time_offset_1     = m.time_offset; // should be 0

    // Iterate all subgrids
    #pragma omp parallel for
    for (int s = 0; s < nr_subgrids; s++) {
        // Load metadata
        const idg::Metadata m  = metadata[s];
        const int offset       = (m.baseline_offset - baseline_offset_1) +
                                 (m.time_offset - time_offset_1);
        const int nr_timesteps = m.nr_timesteps;
        const int aterm_index  = m.aterm_index;
        const int station1     = m.baseline.station1;
        const int station2     = m.baseline.station2;
        const int x_coordinate = m.coordinate.x;
        const int y_coordinate = m.coordinate.y;
        const float w_offset_in_lambda = w_step_in_lambda * m.coordinate.z;
        

        // Compute u and v offset in wavelenghts
        const float u_offset = (x_coordinate + subgrid_size/2 - grid_size/2) * (2*M_PI / image_size);
        const float v_offset = (y_coordinate + subgrid_size/2 - grid_size/2) * (2*M_PI / image_size);
        const float w_offset = 2*M_PI * w_offset_in_lambda;

        // Preload visibilities
        float vis_real[NR_POLARIZATIONS][nr_timesteps*current_nr_channels];
        float vis_imag[NR_POLARIZATIONS][nr_timesteps*current_nr_channels];

        for (int time = 0; time < nr_timesteps; time++) {
            for (int chan = 0; chan < current_nr_channels; chan++) {
                size_t index_src = (offset + time)*nr_channels + (channel_offset + chan);
                size_t index_dst = time * current_nr_channels + chan;
                #pragma vector aligned(vis_real, vis_imag)
                for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                    vis_real[pol][index_dst] = visibilities[index_src][pol].real;
                    vis_imag[pol][index_dst] = visibilities[index_src][pol].imag;
                }
            }
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

        // Iterate all pixels in subgrid
        for (int y = 0; y < subgrid_size; y++) {
            for (int x = 0; x < subgrid_size; x++) {

                // Compute phase
                float phase[nr_timesteps*current_nr_channels];

                // Compute l,m,n
                const float l = (x+0.5-(subgrid_size/2)) * image_size/subgrid_size;
                const float m = (y+0.5-(subgrid_size/2)) * image_size/subgrid_size;
                // evaluate n = 1.0f - sqrt(1.0 - (l * l) - (m * m));
                // accurately for small values of l and m
                const float tmp = (l * l) + (m * m);
                const float n = tmp / (1.0f + sqrtf(1.0f - tmp));

                #pragma vector aligned(uvw_u, uvw_v, uvw_w)
                for (int time = 0; time < nr_timesteps; time++) {
                    // Load UVW coordinates
                    float u = uvw_u[time];
                    float v = uvw_v[time];
                    float w = uvw_w[time];

                    // Compute phase index
                    float phase_index = u*l + v*m + w*n;

                    // Compute phase offset
                    float phase_offset = u_offset*l + v_offset*m + w_offset*n;

                    #pragma vector aligned(wavenumbers, phase)
                    for (int chan = 0; chan < current_nr_channels; chan++) {
                        // Compute phase
                        float wavenumber = wavenumbers[channel_offset + chan];
                        phase[time * current_nr_channels + chan] = phase_offset - (phase_index * wavenumber);
                    }
                } // end time

                #if defined(USE_VML)
                // Compute phasor
                float phasor_real[nr_timesteps*current_nr_channels];
                float phasor_imag[nr_timesteps*current_nr_channels];

                vmsSinCos(
                    nr_timesteps * current_nr_channels,
                    (float *) phase,
                    (float *) phasor_imag,
                    (float *) phasor_real,
                    VML_LA);
                #endif

                // Initialize pixel for every polarization
                float pixels_xx_real = 0.0f;
                float pixels_xy_real = 0.0f;
                float pixels_yx_real = 0.0f;
                float pixels_yy_real = 0.0f;
                float pixels_xx_imag = 0.0f;
                float pixels_xy_imag = 0.0f;
                float pixels_yx_imag = 0.0f;
                float pixels_yy_imag = 0.0f;

                #pragma vector aligned(phase, vis_real, vis_imag)
                #pragma omp simd reduction(+:pixels_xx_real,pixels_xx_imag, \
                                             pixels_xy_real,pixels_xy_imag, \
                                             pixels_yx_real,pixels_yx_imag, \
                                             pixels_yy_real,pixels_yy_imag)
                for (int i = 0; i < nr_timesteps * current_nr_channels; i++) {
                    int time = i / current_nr_channels;
                    int chan = i % current_nr_channels;

                    #if defined(USE_VML)
                    float phasor_real_ = phasor_real[i];
                    float phasor_imag_ = phasor_imag[i];
                    #else
                    float phasor_real_ = cosf(phase[i]);
                    float phasor_imag_ = sinf(phase[i]);
                    #endif

                    pixels_xx_real += vis_real[0][i] * phasor_real_;
                    pixels_xx_imag += vis_real[0][i] * phasor_imag_;
                    pixels_xx_real -= vis_imag[0][i] * phasor_imag_;
                    pixels_xx_imag += vis_imag[0][i] * phasor_real_;

                    pixels_xy_real += vis_real[1][i] * phasor_real_;
                    pixels_xy_imag += vis_real[1][i] * phasor_imag_;
                    pixels_xy_real -= vis_imag[1][i] * phasor_imag_;
                    pixels_xy_imag += vis_imag[1][i] * phasor_real_;

                    pixels_yx_real += vis_real[2][i] * phasor_real_;
                    pixels_yx_imag += vis_real[2][i] * phasor_imag_;
                    pixels_yx_real -= vis_imag[2][i] * phasor_imag_;
                    pixels_yx_imag += vis_imag[2][i] * phasor_real_;

                    pixels_yy_real += vis_real[3][i] * phasor_real_;
                    pixels_yy_imag += vis_real[3][i] * phasor_imag_;
                    pixels_yy_real -= vis_imag[3][i] * phasor_imag_;
                    pixels_yy_imag += vis_imag[3][i] * phasor_real_;
                }

                // Create the pixels
                idg::float2 pixels[NR_POLARIZATIONS];
                pixels[0] = {pixels_xx_real, pixels_xx_imag};
                pixels[1] = {pixels_xy_real, pixels_xy_imag};
                pixels[2] = {pixels_yx_real, pixels_yx_imag};
                pixels[3] = {pixels_yy_real, pixels_yy_imag};

                // Load a term for station1
                idg::float2 aXX1 = aterms[aterm_index * nr_stations + station1][y][x][0];
                idg::float2 aXY1 = aterms[aterm_index * nr_stations + station1][y][x][1];
                idg::float2 aYX1 = aterms[aterm_index * nr_stations + station1][y][x][2];
                idg::float2 aYY1 = aterms[aterm_index * nr_stations + station1][y][x][3];

                // Load aterm for station2
                idg::float2 aXX2 = conj(aterms[aterm_index * nr_stations + station2][y][x][0]);
                idg::float2 aXY2 = conj(aterms[aterm_index * nr_stations + station2][y][x][1]);
                idg::float2 aYX2 = conj(aterms[aterm_index * nr_stations + station2][y][x][2]);
                idg::float2 aYY2 = conj(aterms[aterm_index * nr_stations + station2][y][x][3]);

                apply_aterm(
                    aXX1, aXY1, aYX1, aYY1,
                    aXX2, aXY2, aYX2, aYY2,
                    pixels);

                // Load spheroidal
                float sph = spheroidal[y][x];

                // Compute shifted position in subgrid
                int x_dst = (x + (subgrid_size/2)) % subgrid_size;
                int y_dst = (y + (subgrid_size/2)) % subgrid_size;

                // Set subgrid value
                if (channel_offset == 0) {
                    for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        subgrid[s][pol][y_dst][x_dst] = pixels[pol] * sph;
                    }
                } else {
                    for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        subgrid[s][pol][y_dst][x_dst] += pixels[pol] * sph;
                    }
                }
            } // end x
        } // end y
    } // end s
} // end kernel_gridder_

extern "C" {

void kernel_gridder(
    const int                       nr_subgrids,
    const int                       grid_size,
    const int                       subgrid_size,
    const float                     image_size,
    const float                     w_offset_in_lambda,
    const int                       nr_channels,
    const int                       nr_stations,
    const idg::UVWCoordinate<float> uvw[],
    const float                     wavenumbers[],
    const idg::float2               visibilities[][NR_POLARIZATIONS],
    const float                     spheroidal[subgrid_size][subgrid_size],
    const idg::float2               aterms[][subgrid_size][subgrid_size][NR_POLARIZATIONS],
    const idg::Metadata             metadata[],
          idg::float2               subgrid[][NR_POLARIZATIONS][subgrid_size][subgrid_size])
{
    int channel_offset = 0;

    for (; (channel_offset + 16) <= nr_channels; channel_offset += 16) {
        kernel_gridder_<16>(
            nr_subgrids, grid_size, subgrid_size, image_size, w_offset_in_lambda,
            nr_channels, channel_offset, nr_stations,
            uvw, wavenumbers, visibilities, spheroidal, aterms, metadata, subgrid);
    }

    for (; (channel_offset + 8) <= nr_channels; channel_offset += 8) {
        kernel_gridder_<8>(
            nr_subgrids, grid_size, subgrid_size, image_size, w_offset_in_lambda,
            nr_channels, channel_offset, nr_stations,
            uvw, wavenumbers, visibilities, spheroidal, aterms, metadata, subgrid);
    }

    for (; (channel_offset + 4) <= nr_channels; channel_offset += 4) {
        kernel_gridder_<4>(
            nr_subgrids, grid_size, subgrid_size, image_size, w_offset_in_lambda,
            nr_channels, channel_offset, nr_stations,
            uvw, wavenumbers, visibilities, spheroidal, aterms, metadata, subgrid);
    }

    for (; (channel_offset + 1) <= nr_channels; channel_offset += 1) {
        kernel_gridder_<1>(
            nr_subgrids, grid_size, subgrid_size, image_size, w_offset_in_lambda,
            nr_channels, channel_offset, nr_stations,
            uvw, wavenumbers, visibilities, spheroidal, aterms, metadata, subgrid);
    }
}

} // end extern "C"
