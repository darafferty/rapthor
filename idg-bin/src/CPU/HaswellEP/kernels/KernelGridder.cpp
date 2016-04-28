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

template<int NR_CHANNELS_> void kernel_gridder_(
    const int nr_subgrids,
    const float w_offset,
    const int channel_offset,
    const UVWType		   __restrict__ *uvw,
    const WavenumberType   __restrict__ *wavenumbers,
    const VisibilitiesType __restrict__ *visibilities,
    const SpheroidalType   __restrict__ *spheroidal,
    const ATermType		   __restrict__ *aterm,
    const MetadataType	   __restrict__ *metadata,
    SubGridType			   __restrict__ *subgrid
    )
{
    // Find offset of first subgrid
    const Metadata m            = (*metadata)[0];
    const int baseline_offset_1 = m.baseline_offset;
    const int time_offset_1     = m.time_offset; // should be 0

    // Iterate all subgrids
    #pragma omp parallel for shared(uvw, wavenumbers, visibilities, spheroidal, aterm, metadata) // schedule(dynamic)
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

        // Compute u and v offset in wavelenghts
        const float u_offset = (x_coordinate + SUBGRIDSIZE/2 - GRIDSIZE/2) * (2*M_PI / IMAGESIZE);
        const float v_offset = (y_coordinate + SUBGRIDSIZE/2 - GRIDSIZE/2) * (2*M_PI / IMAGESIZE);

        // Preload visibilities
        float vis_real[nr_timesteps][NR_POLARIZATIONS][NR_CHANNELS_] __attribute__((aligned(32)));
        float vis_imag[nr_timesteps][NR_POLARIZATIONS][NR_CHANNELS_] __attribute__((aligned(32)));

        for (int time = 0; time < nr_timesteps; time++) {
            for (int chan = 0; chan < NR_CHANNELS_; chan++) {
                for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                    vis_real[time][pol][chan] = (*visibilities)[offset + time][chan + channel_offset][pol].real();
                    vis_imag[time][pol][chan] = (*visibilities)[offset + time][chan + channel_offset][pol].imag();
                }
            }
        }

        // Iterate all pixels in subgrid
        for (int y = 0; y < SUBGRIDSIZE; y++) {
            for (int x = 0; x < SUBGRIDSIZE; x++) {
                // Compute phase
                float phase[nr_timesteps][NR_CHANNELS_] __attribute__((aligned(32)));
                compute_phase(
                    nr_timesteps, NR_CHANNELS_,
                    x, y,
                    u_offset, v_offset, w_offset,
                    (float (*)[3]) &uvw[offset][0],
                    (float *) &wavenumbers[channel_offset],
                    phase);

                // Compute phasor
                float phasor_real[nr_timesteps][NR_CHANNELS_] __attribute__((aligned(32)));
                float phasor_imag[nr_timesteps][NR_CHANNELS_] __attribute__((aligned(32)));
                compute_sincos(
                    nr_timesteps * NR_CHANNELS_,
                    (float *) phase,
                    (float *) phasor_imag,
                    (float *) phasor_real);

                // Multiply visibilities with phasor and reduce for all timesteps and channels
                FLOAT_COMPLEX pixels[NR_POLARIZATIONS];
                cmul_reduce(
                    nr_timesteps, NR_CHANNELS_,
                    vis_real, vis_imag,
                    phasor_real, phasor_imag,
                    pixels);

                // Load a term for station1
                FLOAT_COMPLEX aXX1 = (*aterm)[station1][aterm_index][0][y][x];
                FLOAT_COMPLEX aXY1 = (*aterm)[station1][aterm_index][1][y][x];
                FLOAT_COMPLEX aYX1 = (*aterm)[station1][aterm_index][2][y][x];
                FLOAT_COMPLEX aYY1 = (*aterm)[station1][aterm_index][3][y][x];

                // Load aterm for station2
                FLOAT_COMPLEX aXX2 = conj((*aterm)[station2][aterm_index][0][y][x]);
                FLOAT_COMPLEX aXY2 = conj((*aterm)[station2][aterm_index][1][y][x]);
                FLOAT_COMPLEX aYX2 = conj((*aterm)[station2][aterm_index][2][y][x]);
                FLOAT_COMPLEX aYY2 = conj((*aterm)[station2][aterm_index][3][y][x]);

                // Apply aterm
                apply_aterm(
                    aXX1, aXY1, aYX1, aYY1,
                    aXX2, aXY2, aYX2, aYY2,
                    pixels);

                // Load spheroidal
                float sph = (*spheroidal)[y][x];

                // Compute shifted position in subgrid
                int x_dst = (x + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;
                int y_dst = (y + (SUBGRIDSIZE/2)) % SUBGRIDSIZE;

                // Set subgrid value
                for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                    (*subgrid)[s][pol][y_dst][x_dst] = pixels[pol] * sph;
                }
            } // end x
        } // end y
    } // end s
} // end kernel_gridder_

extern "C" {
void kernel_gridder(
    const int nr_subgrids,
    const float w_offset,
    const int nr_channels,
    const UVWType		   __restrict__ *uvw,
    const WavenumberType   __restrict__ *wavenumbers,
    const VisibilitiesType __restrict__ *visibilities,
    const SpheroidalType   __restrict__ *spheroidal,
    const ATermType		   __restrict__ *aterm,
    const MetadataType	   __restrict__ *metadata,
    SubGridType			   __restrict__ *subgrid
    )
{
    int channel_offset = 0;
    for (; (channel_offset + 8) <= nr_channels; channel_offset += 8) {
        kernel_gridder_<8>(
            nr_subgrids, w_offset, channel_offset, uvw, wavenumbers,
            visibilities,spheroidal, aterm, metadata, subgrid);
    }

    for (; channel_offset < nr_channels; channel_offset++) {
        kernel_gridder_<1>(
            nr_subgrids, w_offset, channel_offset, uvw, wavenumbers,
            visibilities,spheroidal, aterm, metadata, subgrid);
    }
} // end kernel_gridder
} // end extern "C"
