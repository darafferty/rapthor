#pragma omp declare target

#include <complex>

#include <math.h>
#include <stdio.h>
#include <immintrin.h>
#include <string.h>
#include <stdint.h>

#include "Types.h"

#define NR_POLARIZATIONS 4

namespace idg {
namespace kernel {
namespace knc {

void degridder(
    const int nr_subgrids,
    const float w_offset,
	const void *_uvw,
	const void *_wavenumbers,
	      void *_visibilities,
	const void *_spheroidal,
	const void *_aterm,
	const void *_metadata,
	const void *_subgrid,
    const int nr_stations,
    const int nr_timesteps,
    const int nr_timeslots,
    const int nr_channels,
    const int gridsize,
    const int subgridsize,
    const float imagesize,
    const int nr_polarizations
	) {
    TYPEDEF_UVW
    TYPEDEF_UVW_TYPE
    TYPEDEF_WAVENUMBER_TYPE
    TYPEDEF_VISIBILITIES_TYPE
    TYPEDEF_SPHEROIDAL_TYPE
    TYPEDEF_ATERM_TYPE
    TYPEDEF_BASELINE
    TYPEDEF_COORDINATE
    TYPEDEF_METADATA
    TYPEDEF_METADATA_TYPE
    TYPEDEF_SUBGRID_TYPE

    UVWType *uvw = (UVWType *) _uvw;
    WavenumberType *wavenumbers = (WavenumberType *) _wavenumbers;
    VisibilitiesType *visibilities = (VisibilitiesType *) _visibilities;
    SpheroidalType *spheroidal = (SpheroidalType *) _spheroidal;
    ATermType *aterm = (ATermType *) _aterm;
    MetadataType *metadata = (MetadataType *) _metadata;
    SubGridType *subgrid = (SubGridType *) _subgrid;

    // Get pointer to visibilities with time and channel dimension
    typedef FLOAT_COMPLEX VisibilityType[nr_timesteps][nr_channels][NR_POLARIZATIONS];
    VisibilityType *vis_ptr = (VisibilityType *) visibilities;

    // Find offset of first subgrid
    const Metadata m = (*metadata)[0];
    const int baseline_offset_1 = m.baseline_offset;
    const int time_offset_1 = m.time_offset; // should be 0

    #pragma omp parallel shared(uvw, wavenumbers, visibilities, spheroidal, aterm, metadata)
    {
        // Iterate all subgrids
        #pragma omp for
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
            FLOAT_COMPLEX pixels[NR_POLARIZATIONS][subgridsize][subgridsize] __attribute__((aligned(32)));
            float pixels_real[NR_POLARIZATIONS][subgridsize][subgridsize] __attribute__((aligned(32)));
            float pixels_imag[NR_POLARIZATIONS][subgridsize][subgridsize] __attribute__((aligned(32)));

            // Apply aterm to subgrid
            for (int y = 0; y < subgridsize; y++) {
                for (int x = 0; x < subgridsize; x++) {
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
                    int x_src = (x + (subgridsize/2)) % subgridsize;
                    int y_src = (y + (subgridsize/2)) % subgridsize;

                    // Load uv values
                    FLOAT_COMPLEX pixelsXX = _spheroidal * (*subgrid)[s][0][y_src][x_src];
                    FLOAT_COMPLEX pixelsXY = _spheroidal * (*subgrid)[s][1][y_src][x_src];
                    FLOAT_COMPLEX pixelsYX = _spheroidal * (*subgrid)[s][2][y_src][x_src];
                    FLOAT_COMPLEX pixelsYY = _spheroidal * (*subgrid)[s][3][y_src][x_src];

                    // Apply aterm to subgrid
                    pixels[0][y][x]  = pixelsXX * aXX1;
                    pixels[0][y][x] += pixelsXY * aYX1;
                    pixels[1][y][x]  = pixelsXX * aXY1;
                    pixels[1][y][x] += pixelsXY * aYY1;
                    pixels[2][y][x]  = pixelsYX * aXX1;
                    pixels[2][y][x] += pixelsYY * aYX1;
                    pixels[3][y][x]  = pixelsYX * aXY1;
                    pixels[3][y][x] += pixelsYY * aYY1;

                    pixelsXX = pixels[0][y][x];
                    pixelsXY = pixels[1][y][x];
                    pixelsYX = pixels[2][y][x];
                    pixelsYY = pixels[3][y][x];
                    pixels[0][y][x]  = pixelsXX * aXX2;
                    pixels[0][y][x] += pixelsYX * aYX2;
                    pixels[1][y][x]  = pixelsXY * aXX2;
                    pixels[1][y][x] += pixelsYY * aYX2;
                    pixels[2][y][x]  = pixelsXX * aXY2;
                    pixels[2][y][x] += pixelsYX * aYY2;
                    pixels[3][y][x]  = pixelsXY * aXY2;
                    pixels[3][y][x] += pixelsYY * aYY2;
                } // end x
            } // end y

            // Split real and imaginary part of pixels
            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                for (int y = 0; y < subgridsize; y++) {
                    for (int x = 0; x < subgridsize; x++) {
                        pixels_real[pol][y][x] = pixels[pol][y][x].real();
                        pixels_imag[pol][y][x] = pixels[pol][y][x].imag();
                    } // end x
                } // end y
            }

            // Compute u and v offset in wavelenghts
            const float u_offset = (x_coordinate + subgridsize/2 - gridsize/2)
                                   * (2*M_PI / imagesize);
            const float v_offset = (y_coordinate + subgridsize/2 - gridsize/2)
                                   * (2*M_PI / imagesize);

            // Iterate all timesteps
            for (int time = 0; time < nr_timesteps; time++) {
                // Load UVW coordinates
                float u = (*uvw)[offset + time].u;
                float v = (*uvw)[offset + time].v;
                float w = (*uvw)[offset + time].w;

                float phase_index[subgridsize][subgridsize] __attribute__((aligned(32)));
                float phase_offset[subgridsize][subgridsize] __attribute__((aligned(32)));

                for (int y = 0; y < subgridsize; y++) {
                    for (int x = 0; x < subgridsize; x++) {
                        // Compute l,m,n
                        const float l = (x-(subgridsize/2)) * imagesize/subgridsize;
                        const float m = (y-(subgridsize/2)) * imagesize/subgridsize;
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
                for (int chan = 0; chan < nr_channels; chan++) {
                    float phase[subgridsize][subgridsize] __attribute__((aligned(32)));
                    float phasor_imag[subgridsize][subgridsize] __attribute__((aligned(32)));
                    float phasor_real[subgridsize][subgridsize] __attribute__((aligned(32)));

                    #pragma nofusion
                    for (int y = 0; y < subgridsize; y++) {
                        for (int x = 0; x < subgridsize; x++) {
                            // Compute phase
                            float wavenumber = (*wavenumbers)[chan];
                            phase[y][x] = phase_offset[y][x] - (phase_index[y][x] * wavenumber);
                        }
                    }

                    #pragma nofusion
                    for (int y = 0; y < subgridsize; y++) {
                        for (int x = 0; x < subgridsize; x++) {
                            // Compute phasor
                            phasor_imag[y][x] = sinf(phase[y][x]);
                            phasor_real[y][x] = cosf(phase[y][x]);
                        }
                    }

                    // Storage for sums
                    float sum_xx_real = 0.0f, sum_xx_imag = 0.0f;
                    float sum_xy_real = 0.0f, sum_xy_imag = 0.0f;
                    float sum_yx_real = 0.0f, sum_yx_imag = 0.0f;
                    float sum_yy_real = 0.0f, sum_yy_imag = 0.0f;

                    #pragma nofusion
                    for (int y = 0; y < subgridsize; y++) {
                        #pragma omp simd reduction(+:\
                            sum_xx_real, sum_xx_imag,\
                            sum_xy_real, sum_xy_imag,\
                            sum_yx_real, sum_yx_imag,\
                            sum_yy_real, sum_yy_imag)
                        for (int x = 0; x < subgridsize; x++) {
                            sum_xx_real +=  phasor_real[y][x] * pixels_real[0][y][x];
                            sum_xx_imag +=  phasor_real[y][x] * pixels_imag[0][y][x];
                            sum_xx_real += -phasor_imag[y][x] * pixels_imag[0][y][x];
                            sum_xx_imag +=  phasor_imag[y][x] * pixels_real[0][y][x];

                            sum_xy_real +=  phasor_real[y][x] * pixels_real[1][y][x];
                            sum_xy_imag +=  phasor_real[y][x] * pixels_imag[1][y][x];
                            sum_xy_real += -phasor_imag[y][x] * pixels_imag[1][y][x];
                            sum_xy_imag +=  phasor_imag[y][x] * pixels_real[1][y][x];

                            sum_yx_real +=  phasor_real[y][x] * pixels_real[2][y][x];
                            sum_yx_imag +=  phasor_real[y][x] * pixels_imag[2][y][x];
                            sum_yx_real += -phasor_imag[y][x] * pixels_imag[2][y][x];
                            sum_yx_imag +=  phasor_imag[y][x] * pixels_real[2][y][x];

                            sum_yy_real +=  phasor_real[y][x] * pixels_real[3][y][x];
                            sum_yy_imag +=  phasor_real[y][x] * pixels_imag[3][y][x];
                            sum_yy_real += -phasor_imag[y][x] * pixels_imag[3][y][x];
                            sum_yy_imag +=  phasor_imag[y][x] * pixels_real[3][y][x];
                        }
                    }

                    // Store visibilities
                    const float scale = 1.0f / (subgridsize*subgridsize);
                    (*vis_ptr)[offset + time][chan][0] = {scale*sum_xx_real, scale*sum_xx_imag};
                    (*vis_ptr)[offset + time][chan][1] = {scale*sum_xy_real, scale*sum_xy_imag};
                    (*vis_ptr)[offset + time][chan][2] = {scale*sum_yx_real, scale*sum_yx_imag};
                    (*vis_ptr)[offset + time][chan][3] = {scale*sum_yy_real, scale*sum_yy_imag};
                } // end for channel
            } // end for time
        } // end for subgrids
    } // pragma parallel
} // end degridder

} // end namespace knc
} // end namespace kernel
} // end namespace idg

#pragma omp end declare target
