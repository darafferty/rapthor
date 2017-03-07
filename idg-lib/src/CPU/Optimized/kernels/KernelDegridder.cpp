#include <cmath>

#if defined(__INTEL_COMPILER) || defined(HAVE_MKL)
#define USE_VML
#define VML_PRECISION VML_LA
#include <mkl_vml.h>
#endif

#include "Types.h"
#include "Math.h"

template<int current_nr_channels>
void kernel_degridder_(
    const int                       nr_subgrids,
    const int                       grid_size,
    const float                     image_size,
    const float                     w_offset_in_lambda,
    const int                       nr_channels,
    const int                       channel_offset,
    const int                       nr_stations,
    const idg::UVWCoordinate<float> uvw[],
    const float                     wavenumbers[],
          idg::float2               visibilities[][NR_POLARIZATIONS],
    const float                     spheroidal[SUBGRIDSIZE][SUBGRIDSIZE],
    const idg::float2               aterms[][SUBGRIDSIZE][SUBGRIDSIZE][NR_POLARIZATIONS],
    const idg::Metadata             metadata[],
    const idg::float2               subgrid[][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE]
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
        const int offset       = (m.baseline_offset - baseline_offset_1)
                                 + (m.time_offset - time_offset_1);
        const int nr_timesteps = m.nr_timesteps;
        const int aterm_index  = m.aterm_index;
        const int station1     = m.baseline.station1;
        const int station2     = m.baseline.station2;
        const int x_coordinate = m.coordinate.x;
        const int y_coordinate = m.coordinate.y;

        // Storage
        float pixels_real[NR_POLARIZATIONS][SUBGRIDSIZE*SUBGRIDSIZE] __attribute__((aligned(64)));
        float pixels_imag[NR_POLARIZATIONS][SUBGRIDSIZE*SUBGRIDSIZE] __attribute__((aligned(64)));

        // Apply aterm to subgrid
        for (int i = 0; i < SUBGRIDSIZE * SUBGRIDSIZE; i++) {
            int y = i / SUBGRIDSIZE;
            int x = i % SUBGRIDSIZE;

            // Load aterm for station1
            idg::float2 aXX1 = aterms[aterm_index * nr_stations + station1][y][x][0];
            idg::float2 aXY1 = aterms[aterm_index * nr_stations + station1][y][x][1];
            idg::float2 aYX1 = aterms[aterm_index * nr_stations + station1][y][x][2];
            idg::float2 aYY1 = aterms[aterm_index * nr_stations + station1][y][x][3];

            // Load aterm for station2
            idg::float2 aXX2 = conj(aterms[aterm_index * nr_stations + station2][y][x][0]);
            idg::float2 aXY2 = conj(aterms[aterm_index * nr_stations + station2][y][x][1]);
            idg::float2 aYX2 = conj(aterms[aterm_index * nr_stations + station2][y][x][2]);
            idg::float2 aYY2 = conj(aterms[aterm_index * nr_stations + station2][y][x][3]);

            // Load spheroidal
            float _spheroidal = spheroidal[y][x];

            // Compute shifted position in subgrid
            int x_src = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
            int y_src = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

            // Load pixel values and apply spheroidal
            idg::float2 pixels[NR_POLARIZATIONS];
            pixels[0] = _spheroidal * subgrid[s][0][y_src][x_src];
            pixels[1] = _spheroidal * subgrid[s][1][y_src][x_src];
            pixels[2] = _spheroidal * subgrid[s][2][y_src][x_src];
            pixels[3] = _spheroidal * subgrid[s][3][y_src][x_src];

            apply_aterm(
                aXX1, aXY1, aYX1, aYY1,
                aXX2, aXY2, aYX2, aYY2,
                pixels);

            // Store pixels
            pixels_real[0][i] = pixels[0].real;
            pixels_real[1][i] = pixels[1].real;
            pixels_real[2][i] = pixels[2].real;
            pixels_real[3][i] = pixels[3].real;
            pixels_imag[0][i] = pixels[0].imag;
            pixels_imag[1][i] = pixels[1].imag;
            pixels_imag[2][i] = pixels[2].imag;
            pixels_imag[3][i] = pixels[3].imag;
        }

        // Compute u and v offset in wavelenghts
        const float u_offset = (x_coordinate + SUBGRIDSIZE/2 - grid_size/2)
                               * (2*M_PI / image_size);
        const float v_offset = (y_coordinate + SUBGRIDSIZE/2 - grid_size/2)
                               * (2*M_PI / image_size);
        const float w_offset = 2*M_PI * w_offset_in_lambda;

        // Iterate all timesteps
        for (int time = 0; time < nr_timesteps; time++) {
            // Load UVW coordinates
            float u = uvw[offset + time].u;
            float v = uvw[offset + time].v;
            float w = uvw[offset + time].w;

            float phase_index[SUBGRIDSIZE*SUBGRIDSIZE] __attribute__((aligned(32)));
            float phase_offset[SUBGRIDSIZE*SUBGRIDSIZE] __attribute__((aligned(32)));

            for (int i = 0; i < SUBGRIDSIZE * SUBGRIDSIZE; i++) {
                int y = i / SUBGRIDSIZE;
                int x = i % SUBGRIDSIZE;

                // Compute l,m,n
                const float l = (x+0.5-(SUBGRIDSIZE/2)) * image_size/SUBGRIDSIZE;
                const float m = (y+0.5-(SUBGRIDSIZE/2)) * image_size/SUBGRIDSIZE;
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
                float phase[SUBGRIDSIZE*SUBGRIDSIZE] __attribute__((aligned(64)));

                for (int i = 0; i < SUBGRIDSIZE * SUBGRIDSIZE; i++) {
                    int y = i / SUBGRIDSIZE;
                    int x = i % SUBGRIDSIZE;

                    // Compute phase
                    float wavenumber = wavenumbers[channel_offset + chan];
                    phase[i] = (phase_index[i] * wavenumber) - phase_offset[i];
                }

                #if defined(USE_VML)
                // Compute phasor
                float phasor_real[SUBGRIDSIZE*SUBGRIDSIZE] __attribute__((aligned(64)));
                float phasor_imag[SUBGRIDSIZE*SUBGRIDSIZE] __attribute__((aligned(64)));

                vmsSinCos(
                    SUBGRIDSIZE*SUBGRIDSIZE,
                    (float *) phase,
                    (float *) phasor_imag,
                    (float *) phasor_real,
                    VML_LA);
                #endif

                // Multiply phasor with pixels and reduce for all pixels
                idg::float2 sums[NR_POLARIZATIONS];

                // Initialize pixel for every polarization
                float sums_xx_real = 0.0f;
                float sums_xy_real = 0.0f;
                float sums_yx_real = 0.0f;
                float sums_yy_real = 0.0f;
                float sums_xx_imag = 0.0f;
                float sums_xy_imag = 0.0f;
                float sums_yx_imag = 0.0f;
                float sums_yy_imag = 0.0f;

                // Accumulate visibility value from all pixels
                #pragma omp simd reduction(+:sums_xx_real,sums_xx_imag, \
                                             sums_xy_real,sums_xy_imag, \
                                             sums_yx_real,sums_yx_imag, \
                                             sums_yy_real,sums_yy_imag)
                for (int i = 0; i < SUBGRIDSIZE * SUBGRIDSIZE; i++) {
                    #if defined(USE_VML)
                    float phasor_real_ = phasor_real[i];
                    float phasor_imag_ = phasor_imag[i];
                    #else
                    float phasor_real_ = cosf(phase[i]);
                    float phasor_imag_ = sinf(phase[i]);
                    #endif

                    sums_xx_real += phasor_real_ * pixels_real[0][i];
                    sums_xx_imag += phasor_real_ * pixels_imag[0][i];
                    sums_xx_real -= phasor_imag_ * pixels_imag[0][i];
                    sums_xx_imag += phasor_imag_ * pixels_real[0][i];

                    sums_xy_real += phasor_real_ * pixels_real[1][i];
                    sums_xy_imag += phasor_real_ * pixels_imag[1][i];
                    sums_xy_real -= phasor_imag_ * pixels_imag[1][i];
                    sums_xy_imag += phasor_imag_ * pixels_real[1][i];

                    sums_yx_real += phasor_real_ * pixels_real[2][i];
                    sums_yx_imag += phasor_real_ * pixels_imag[2][i];
                    sums_yx_real -= phasor_imag_ * pixels_imag[2][i];
                    sums_yx_imag += phasor_imag_ * pixels_real[2][i];

                    sums_yy_real += phasor_real_ * pixels_real[3][i];
                    sums_yy_imag += phasor_real_ * pixels_imag[3][i];
                    sums_yy_real -= phasor_imag_ * pixels_imag[3][i];
                    sums_yy_imag += phasor_imag_ * pixels_real[3][i];
                }

                // Combine real and imaginary parts
                sums[0] = {sums_xx_real, sums_xx_imag};
                sums[1] = {sums_xy_real, sums_xy_imag};
                sums[2] = {sums_yx_real, sums_yx_imag};
                sums[3] = {sums_yy_real, sums_yy_imag};

                // Store visibilities
                const float scale = 1.0f / (SUBGRIDSIZE*SUBGRIDSIZE);
                size_t index = (offset + time)*nr_channels + (channel_offset + chan);
                visibilities[index][0] = {scale*sums[0].real, scale*sums[0].imag};
                visibilities[index][1] = {scale*sums[1].real, scale*sums[1].imag};
                visibilities[index][2] = {scale*sums[2].real, scale*sums[2].imag};
                visibilities[index][3] = {scale*sums[3].real, scale*sums[3].imag};
            } // end for channel
        } // end for time
    } // end #pragma parallel
} // end kernel_degridder_

extern "C" {
void kernel_degridder(
    const int           nr_subgrids,
    const int           gridsize,
    const float         imagesize,
    const float         w_offset,
    const int           nr_channels,
    const int           nr_stations,
    const idg::UVWCoordinate<float>	uvw[],
    const float                     wavenumbers[],
          idg::float2               visibilities[][NR_POLARIZATIONS],
    const float                     spheroidal[SUBGRIDSIZE][SUBGRIDSIZE],
    const idg::float2               aterm[][SUBGRIDSIZE][SUBGRIDSIZE][NR_POLARIZATIONS],
    const idg::Metadata             metadata[],
    const idg::float2               subgrid[][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE]
    )
{
    int channel_offset = 0;
    for (; (channel_offset + 16) <= nr_channels; channel_offset += 16) {
        kernel_degridder_<16>(
            nr_subgrids, gridsize, imagesize, w_offset, nr_channels, channel_offset, nr_stations,
            uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrid);
    }

    for (; (channel_offset + 8) <= nr_channels; channel_offset += 8) {
        kernel_degridder_<8>(
            nr_subgrids, gridsize, imagesize, w_offset, nr_channels, channel_offset, nr_stations,
            uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrid);
    }

    for (; (channel_offset + 4) <= nr_channels; channel_offset += 4) {
        kernel_degridder_<4>(
            nr_subgrids, gridsize, imagesize, w_offset, nr_channels, channel_offset, nr_stations,
            uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrid);
    }

    for (; channel_offset < nr_channels; channel_offset++) {
        kernel_degridder_<1>(
            nr_subgrids, gridsize, imagesize, w_offset, nr_channels, channel_offset, nr_stations,
            uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrid);
    }
} // end kernel_degridder
} // end extern "C"
