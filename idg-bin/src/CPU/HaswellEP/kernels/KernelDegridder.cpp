#include <complex>
#include <cmath>
#include <cstring>
#include <omp.h>

#if defined(__INTEL_COMPILER)
#define USE_VML
#define VML_PRECISION VML_LA
#include <mkl_vml.h>
#endif

#include "Types.h"
#include "Math.h"

template<int NR_CHANNELS_> void kernel_degridder_(
    const int nr_subgrids,
    const float w_offset,
    const int channel_offset,
    const UVWType		 __restrict__ *uvw,
    const WavenumberType __restrict__ *wavenumbers,
    VisibilitiesType	 __restrict__ *visibilities,
    const SpheroidalType __restrict__ *spheroidal,
    const ATermType		 __restrict__ *aterm,
    const MetadataType	 __restrict__ *metadata,
    const SubGridType	 __restrict__ *subgrid
    )
{
    // Find offset of first subgrid
    const Metadata m = (*metadata)[0];
    const int baseline_offset_1 = m.baseline_offset;
    const int time_offset_1     = m.time_offset; // should be 0

    // Iterate all subgrids
    #pragma omp parallel for shared(uvw, wavenumbers, visibilities, spheroidal, aterm, metadata)
    for (int s = 0; s < nr_subgrids; s++) {

        // Load metadata
        const Metadata m = (*metadata)[s];
        const int offset = (m.baseline_offset - baseline_offset_1)
              + (m.time_offset - time_offset_1);
        const int nr_timesteps = m.nr_timesteps;
        const int aterm_index = m.aterm_index;
        const int station1 = m.baseline.station1;
        const int station2 = m.baseline.station2;
        const int x_coordinate = m.coordinate.x;
        const int y_coordinate = m.coordinate.y;

        // Storage
        FLOAT_COMPLEX pixels[NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));
        float pixels_real[NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));
        float pixels_imag[NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));

        // Apply aterm to subgrid
        for (int y = 0; y < SUBGRIDSIZE; y++) {
            for (int x = 0; x < SUBGRIDSIZE; x++) {
                // Load aterm for station1
                FLOAT_COMPLEX aXX1 = (*aterm)[station1][aterm_index][0][y][x];
                FLOAT_COMPLEX aXY1 = (*aterm)[station1][aterm_index][1][y][x];
                FLOAT_COMPLEX aYX1 = (*aterm)[station1][aterm_index][2][y][x];
                FLOAT_COMPLEX aYY1 = (*aterm)[station1][aterm_index][3][y][x];

                // Load aterm for station2
                FLOAT_COMPLEX aXX2 = conj((*aterm)[station2][aterm_index][0][y][x]);
                FLOAT_COMPLEX aXY2 = conj((*aterm)[station2][aterm_index][1][y][x]);
                FLOAT_COMPLEX aYX2 = conj((*aterm)[station2][aterm_index][2][y][x]);
                FLOAT_COMPLEX aYY2 = conj((*aterm)[station2][aterm_index][3][y][x]);

                // Load spheroidal
                float _spheroidal = (*spheroidal)[y][x];

                // Compute shifted position in subgrid
                int x_src = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                int y_src = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

                // Load pixel values and apply spheroidal
                FLOAT_COMPLEX pixels_[NR_POLARIZATIONS];
                pixels_[0] = _spheroidal * (*subgrid)[s][0][y_src][x_src];
                pixels_[1] = _spheroidal * (*subgrid)[s][1][y_src][x_src];
                pixels_[2] = _spheroidal * (*subgrid)[s][2][y_src][x_src];
                pixels_[3] = _spheroidal * (*subgrid)[s][3][y_src][x_src];

                // Apply aterm
                apply_aterm(
                    aXX1, aXY1, aYX1, aYY1,
                    aXX2, aXY2, aYX2, aYY2,
                    pixels_);

                // Store pixels
                pixels[0][y][x] = pixels_[0];
                pixels[1][y][x] = pixels_[1];
                pixels[2][y][x] = pixels_[2];
                pixels[3][y][x] = pixels_[3];
            } // end x
        } // end y

        // Split real and imaginary part of pixels
        for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
            for (int y = 0; y < SUBGRIDSIZE; y++) {
                for (int x = 0; x < SUBGRIDSIZE; x++) {
                    pixels_real[pol][y][x] = pixels[pol][y][x].real();
                    pixels_imag[pol][y][x] = pixels[pol][y][x].imag();
                } // end x
            } // end y
        }

        // Compute u and v offset in wavelenghts
        const float u_offset = (x_coordinate + SUBGRIDSIZE/2 - GRIDSIZE/2)
                               * (2*M_PI / IMAGESIZE);
        const float v_offset = (y_coordinate + SUBGRIDSIZE/2 - GRIDSIZE/2)
                               * (2*M_PI / IMAGESIZE);

        // Iterate all timesteps
        for (int time = 0; time < nr_timesteps; time++) {
            // Load UVW coordinates
            float u = (*uvw)[offset + time].u;
            float v = (*uvw)[offset + time].v;
            float w = (*uvw)[offset + time].w;

            float phase_index[SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));
            float phase_offset[SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));

            for (int y = 0; y < SUBGRIDSIZE; y++) {
                for (int x = 0; x < SUBGRIDSIZE; x++) {
                    // Compute l,m,n
                    const float l = (x-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
                    const float m = (y-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
                    // evaluate n = 1.0f - sqrt(1.0 - (l * l) - (m * m));
                    // accurately for small values of l and m
                    const float tmp = (l * l) + (m * m);
                    const float n = tmp / (1.0f + sqrtf(1.0f - tmp));

                    // Compute phase index
                    phase_index[y][x] = u*l + v*m + w*n;

                    // Compute phase offset
                    phase_offset[y][x] = u_offset*l + v_offset*m + w_offset*n;
                }
            }

            // Iterate all channels
            for (int chan = 0; chan < NR_CHANNELS_; chan++) {

                float phase[SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));
                float phasor_imag[SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));
                float phasor_real[SUBGRIDSIZE][SUBGRIDSIZE] __attribute__((aligned(32)));

                #pragma nofusion
                for (int y = 0; y < SUBGRIDSIZE; y++) {
                    for (int x = 0; x < SUBGRIDSIZE; x++) {
                        // Compute phase
                        float wavenumber = (*wavenumbers)[chan];
                        phase[y][x] = phase_offset[y][x] - (phase_index[y][x] * wavenumber);
                    }
                }

                // Compute phasor
                compute_sincos(
                    SUBGRIDSIZE * SUBGRIDSIZE,
                    (float *) phase,
                    (float *) phasor_imag,
                    (float *) phasor_real);


                // Multiply phasor with pixels and reduce for all pixels
                FLOAT_COMPLEX sums[NR_POLARIZATIONS];
                cmul_reduce_degridder(
                    phasor_real, phasor_imag,
                    pixels_real, pixels_imag,
                    sums);

                // Store visibilities
                const float scale = 1.0f / (SUBGRIDSIZE*SUBGRIDSIZE);
                (*visibilities)[offset + time][chan][0] = {scale*sums[0].real(), scale*sums[0].imag()};
                (*visibilities)[offset + time][chan][1] = {scale*sums[1].real(), scale*sums[1].imag()};
                (*visibilities)[offset + time][chan][2] = {scale*sums[2].real(), scale*sums[2].imag()};
                (*visibilities)[offset + time][chan][3] = {scale*sums[3].real(), scale*sums[3].imag()};
            } // end for channel
        } // end for time
    } // end #pragma parallel
}

extern "C" {
void kernel_degridder(
    const int nr_subgrids,
    const float w_offset,
    const int nr_channels,
    const UVWType		 __restrict__ *uvw,
    const WavenumberType __restrict__ *wavenumbers,
    VisibilitiesType	 __restrict__ *visibilities,
    const SpheroidalType __restrict__ *spheroidal,
    const ATermType		 __restrict__ *aterm,
    const MetadataType	 __restrict__ *metadata,
    const SubGridType	 __restrict__ *subgrid

    )
{
    int channel_offset = 0;
    for (; (channel_offset + 8) <= nr_channels; channel_offset += 8) {
        kernel_degridder_<8>(
            nr_subgrids, w_offset, channel_offset, uvw, wavenumbers,
            visibilities,spheroidal, aterm, metadata, subgrid);
    }

    for (; channel_offset < nr_channels; channel_offset++) {
        kernel_degridder_<1>(
            nr_subgrids, w_offset, channel_offset, uvw, wavenumbers,
            visibilities,spheroidal, aterm, metadata, subgrid);
    }
} // end kernel_degridder
} // end extern "C"
