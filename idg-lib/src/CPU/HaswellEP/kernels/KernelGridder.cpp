#include <complex>
#include <cmath>
#include <cstring>
#include <omp.h>

#include <stdio.h>

#if defined(__INTEL_COMPILER) || defined(HAVE_MKL)
#define USE_VML
#define VML_PRECISION VML_LA
#include <mkl_vml.h>
#endif

#include "Types.h"
#include "Math.h"

template<int current_nr_channels>
void kernel_gridder_(
    const int           nr_subgrids,
    const float         w_offset_in_lambda,
    const int           nr_channels,
    const int           channel_offset,
    const int           nr_stations,
    const idg::UVW		uvw[],
    const float         wavenumbers[],
    const idg::float2   visibilities[][NR_POLARIZATIONS],
    const float         spheroidal[SUBGRIDSIZE][SUBGRIDSIZE],
    const idg::float2   aterm[][SUBGRIDSIZE][SUBGRIDSIZE][NR_POLARIZATIONS],
    const idg::Metadata metadata[],
          idg::float2   subgrid[][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE])
{
    // Find offset of first subgrid
    const idg::Metadata m       = metadata[0];
    const int baseline_offset_1 = m.baseline_offset;
    const int time_offset_1     = m.time_offset; // should be 0

    // Iterate all subgrids
    #pragma omp parallel for shared(uvw, wavenumbers, visibilities, spheroidal, aterm, metadata)
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

        // Compute u and v offset in wavelenghts
        const float u_offset = (x_coordinate + SUBGRIDSIZE/2 - GRIDSIZE/2) * (2*M_PI / IMAGESIZE);
        const float v_offset = (y_coordinate + SUBGRIDSIZE/2 - GRIDSIZE/2) * (2*M_PI / IMAGESIZE);
        const float w_offset = 2*M_PI * w_offset_in_lambda; // TODO: check!

        // Preload visibilities
        float vis_real[nr_timesteps][NR_POLARIZATIONS][current_nr_channels] __attribute__((aligned(32)));
        float vis_imag[nr_timesteps][NR_POLARIZATIONS][current_nr_channels] __attribute__((aligned(32)));

        for (int time = 0; time < nr_timesteps; time++) {
            for (int chan = 0; chan < current_nr_channels; chan++) {
                size_t index = (offset + time)*nr_channels + (channel_offset + chan);
                for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                    vis_real[time][pol][chan] = visibilities[index][pol].real;
                    vis_imag[time][pol][chan] = visibilities[index][pol].imag;
                }
            }
        }

        // Iterate all pixels in subgrid
        for (int y = 0; y < SUBGRIDSIZE; y++) {
            for (int x = 0; x < SUBGRIDSIZE; x++) {

                // Compute phase
                float phase[nr_timesteps][current_nr_channels] __attribute__((aligned(32)));

                // Compute l,m,n
                const float l = (x-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
                const float m = (y-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
                // evaluate n = 1.0f - sqrt(1.0 - (l * l) - (m * m));
                // accurately for small values of l and m
                const float tmp = (l * l) + (m * m);
                const float n = tmp / (1.0f + sqrtf(1.0f - tmp));

                for (int time = 0; time < nr_timesteps; time++) {
                    // Load UVW coordinates
                    float u = uvw[offset + time].u;
                    float v = uvw[offset + time].v;
                    float w = uvw[offset + time].w;

                    // Compute phase index
                    float phase_index = u*l + v*m + w*n;

                    // Compute phase offset
                    float phase_offset = u_offset*l + v_offset*m + w_offset*n;

                    for (int chan = 0; chan < current_nr_channels; chan++) {
                        // Compute phase
                        float wavenumber = wavenumbers[channel_offset + chan];
                        phase[time][chan] = phase_offset - (phase_index * wavenumber);
                    }
                } // end time

                // Compute phasor
                float phasor_real[nr_timesteps][current_nr_channels] __attribute__((aligned(32)));
                float phasor_imag[nr_timesteps][current_nr_channels] __attribute__((aligned(32)));
                compute_sincos(
                    nr_timesteps * current_nr_channels,
                    (float *) phase,
                    (float *) phasor_imag,
                    (float *) phasor_real);

                // Multiply visibilities with phasor and reduce for all timesteps and channels
                idg::float2 pixels[NR_POLARIZATIONS];

                // Initialize pixel for every polarization
                float pixels_xx_real = 0.0f;
                float pixels_xy_real = 0.0f;
                float pixels_yx_real = 0.0f;
                float pixels_yy_real = 0.0f;
                float pixels_xx_imag = 0.0f;
                float pixels_xy_imag = 0.0f;
                float pixels_yx_imag = 0.0f;
                float pixels_yy_imag = 0.0f;

                // Update pixel for every timestep
                for (int time = 0; time < nr_timesteps; time++) {
                     // Update pixel for every channel
                     #pragma omp simd reduction(+:pixels_xx_real,pixels_xx_imag,  \
                                                  pixels_xy_real,pixels_xy_imag,  \
                                                  pixels_yx_real,pixels_yx_imag,  \
                                                  pixels_yy_real,pixels_yy_imag)
                     for (int chan = 0; chan < current_nr_channels; chan++) {
                          pixels_xx_real += vis_real[time][0][chan] * phasor_real[time][chan];
                          pixels_xx_imag += vis_real[time][0][chan] * phasor_imag[time][chan];
                          pixels_xx_real -= vis_imag[time][0][chan] * phasor_imag[time][chan];
                          pixels_xx_imag += vis_imag[time][0][chan] * phasor_real[time][chan];

                          pixels_xy_real += vis_real[time][1][chan] * phasor_real[time][chan];
                          pixels_xy_imag += vis_real[time][1][chan] * phasor_imag[time][chan];
                          pixels_xy_real -= vis_imag[time][1][chan] * phasor_imag[time][chan];
                          pixels_xy_imag += vis_imag[time][1][chan] * phasor_real[time][chan];

                          // #pragma distribute_point

                          pixels_yx_real += vis_real[time][2][chan] * phasor_real[time][chan];
                          pixels_yx_imag += vis_real[time][2][chan] * phasor_imag[time][chan];
                          pixels_yx_real -= vis_imag[time][2][chan] * phasor_imag[time][chan];
                          pixels_yx_imag += vis_imag[time][2][chan] * phasor_real[time][chan];

                          pixels_yy_real += vis_real[time][3][chan] * phasor_real[time][chan];
                          pixels_yy_imag += vis_real[time][3][chan] * phasor_imag[time][chan];
                          pixels_yy_real -= vis_imag[time][3][chan] * phasor_imag[time][chan];
                          pixels_yy_imag += vis_imag[time][3][chan] * phasor_real[time][chan];
                     }
                 }

                // Combine real and imaginary parts
                pixels[0] = {pixels_xx_real, pixels_xx_imag};
                pixels[1] = {pixels_xy_real, pixels_xy_imag};
                pixels[2] = {pixels_yx_real, pixels_yx_imag};
                pixels[3] = {pixels_yy_real, pixels_yy_imag};

                // Load a term for station1
                idg::float2 aXX1 = aterm[aterm_index * nr_stations + station1][y][x][0];
                idg::float2 aXY1 = aterm[aterm_index * nr_stations + station1][y][x][1];
                idg::float2 aYX1 = aterm[aterm_index * nr_stations + station1][y][x][2];
                idg::float2 aYY1 = aterm[aterm_index * nr_stations + station1][y][x][3];

                // Load aterm for station2
                idg::float2 aXX2 = aterm[aterm_index * nr_stations + station2][y][x][0];
                idg::float2 aXY2 = aterm[aterm_index * nr_stations + station2][y][x][1];
                idg::float2 aYX2 = aterm[aterm_index * nr_stations + station2][y][x][2];
                idg::float2 aYY2 = aterm[aterm_index * nr_stations + station2][y][x][3];

                // Apply aterm
                apply_aterm(
                    aXX1, aXY1, aYX1, aYY1,
                    aXX2, aXY2, aYX2, aYY2,
                    pixels);

                // Load spheroidal
                float sph = spheroidal[y][x];

                // Compute shifted position in subgrid
                int x_dst = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                int y_dst = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

                // Set subgrid value
                if (channel_offset==0) {
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
    const int           nr_subgrids,
    const float         w_offset,
    const int           nr_channels,
    const int           nr_stations,
    const idg::UVW		uvw[],
    const float         wavenumbers[],
    const idg::float2   visibilities[][NR_POLARIZATIONS],
    const float         spheroidal[SUBGRIDSIZE][SUBGRIDSIZE],
    const idg::float2   aterm[][SUBGRIDSIZE][SUBGRIDSIZE][NR_POLARIZATIONS],
    const idg::Metadata metadata[],
          idg::float2   subgrid[][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE]
    )
{
    int channel_offset = 0;
    for (; (channel_offset + 8) <= nr_channels; channel_offset += 8) {
        kernel_gridder_<8>(
            nr_subgrids, w_offset, nr_channels, channel_offset, nr_stations,
            uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrid);
    }

    for (; (channel_offset + 4) <= nr_channels; channel_offset += 4) {
        kernel_gridder_<4>(
            nr_subgrids, w_offset, nr_channels, channel_offset, nr_stations,
            uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrid);
    }

    for (; channel_offset < nr_channels; channel_offset++) {
        kernel_gridder_<1>(
            nr_subgrids, w_offset, nr_channels, channel_offset, nr_stations,
            uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrid);
    }
} // end kernel_gridder

} // end extern "C"
