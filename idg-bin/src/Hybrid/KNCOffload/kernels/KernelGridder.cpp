#pragma omp declare target
#include <complex>

#include <math.h>
#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
#include <string.h>
#include <stdint.h>

#include "Types.h"

namespace idg {

void kernel_gridder (
	const int jobsize, const float w_offset,
	const void *_uvw,
	const void *_wavenumbers,
	const void *_visibilities,
	const void *_spheroidal,
	const void *_aterm,
	const void *_metadata,
	void	   *_subgrid,
    const int nr_stations,
    const int nr_timesteps,
    const int nr_timeslots,
    const int nr_channels,
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

    #pragma omp parallel shared(uvw, wavenumbers, visibilities, spheroidal, aterm, metadata)
    {
    // Iterate all subgrids
    #pragma omp for
	for (int s = 0; s < jobsize; s++) {
        // Load metadata
        const Metadata m = (*metadata)[0];
        int time_nr = m.time_nr;
        int station1 = m.baseline.station1;
        int station2 = m.baseline.station2;
        int x_coordinate = m.coordinate.x;
        int y_coordinate = m.coordinate.y;

        // Compute u and v offset in wavelenghts
        float u_offset = (x_coordinate + subgridsize/2) / imagesize;
        float v_offset = (y_coordinate + subgridsize/2) / imagesize;

        // Iterate all pixels in subgrid
        for (int y = 0; y < subgridsize; y++) {
            for (int x = 0; x < subgridsize; x++) {
                // Initialize pixel for every polarization
                FLOAT_COMPLEX pixels[nr_polarizations];
                memset(pixels, 0, nr_polarizations * sizeof(FLOAT_COMPLEX));

                // Compute l,m,n
                float l = -(x-(subgridsize/2)) * imagesize/subgridsize;
                float m =  (y-(subgridsize/2)) * imagesize/subgridsize;
                float n = 1.0f - (float) sqrt(1.0 - (double) (l * l) - (double) (m * m));
 
                // Iterate all timesteps
                for (int time = 0; time < nr_timesteps; time++) {
                    // Load UVW coordinates
                    float u = (*uvw)[s][time].u;
                    float v = (*uvw)[s][time].v;
                    float w = (*uvw)[s][time].w;

                    // Compute phase index
                    float phase_index = u*l + v*m + w*n;

                    // Compute phase offset
                    float phase_offset = u_offset*l + v_offset*m + w_offset*n;

                    // Update pixel for every channel
                    for (int chan = 0; chan < nr_channels; chan++) {
                        // Compute phase
                        float wavenumber = (*wavenumbers)[chan];
                        float phase  = (phase_index * wavenumber) - phase_offset;

                        // Compute phasor
                        float phasor_real = cosf(phase);
                        float phasor_imag = sinf(phase);
                        FLOAT_COMPLEX phasor = FLOAT_COMPLEX(phasor_real, phasor_imag);

                        // Update pixel for every polarization
                        for (int pol = 0; pol < nr_polarizations; pol++) {
                            FLOAT_COMPLEX visibility = (*visibilities)[s][time][chan][pol];
                            pixels[pol] += visibility * phasor;
                        }
                    }
                }
                
                // Load a term for station1
                FLOAT_COMPLEX aXX1 = (*aterm)[station1][time_nr][0][y][x];
                FLOAT_COMPLEX aXY1 = (*aterm)[station1][time_nr][1][y][x];
                FLOAT_COMPLEX aYX1 = (*aterm)[station1][time_nr][2][y][x];
                FLOAT_COMPLEX aYY1 = (*aterm)[station1][time_nr][3][y][x];

                // Load aterm for station2
                FLOAT_COMPLEX aXX2 = conj((*aterm)[station2][time_nr][0][y][x]);
                FLOAT_COMPLEX aXY2 = conj((*aterm)[station2][time_nr][1][y][x]);
                FLOAT_COMPLEX aYX2 = conj((*aterm)[station2][time_nr][2][y][x]);
                FLOAT_COMPLEX aYY2 = conj((*aterm)[station2][time_nr][3][y][x]);

                // Apply aterm to subgrid
                pixels[0]  = (pixels[0] * aXX1);
                pixels[0] += (pixels[1] * aYX1);
                pixels[0] += (pixels[0] * aXX2);
                pixels[0] += (pixels[2] * aYX2);
                pixels[1]  = (pixels[0] * aXY1);
                pixels[1] += (pixels[1] * aYY1);
                pixels[1] += (pixels[1] * aXX2);
                pixels[1] += (pixels[3] * aYX2);
                pixels[2]  = (pixels[2] * aXX1);
                pixels[2] += (pixels[3] * aYX1);
                pixels[2] += (pixels[0] * aXY2);
                pixels[2] += (pixels[2] * aYY2);
                pixels[3]  = (pixels[2] * aXY1);
                pixels[3] += (pixels[3] * aYY1);
                pixels[3] += (pixels[1] * aXY2);
                pixels[3] += (pixels[3] * aYY2);

                // Load spheroidal
                float sph = (*spheroidal)[y][x];

                // Compute shifted position in subgrid
                int x_dst = (x + (subgridsize/2)) % subgridsize;
                int y_dst = (y + (subgridsize/2)) % subgridsize;

                // Set subgrid value
                for (int pol = 0; pol < nr_polarizations; pol++) {
                    (*subgrid)[s][pol][y_dst][x_dst] = pixels[pol] * sph;
                }
            }
        }
    }
    }
}

uint64_t kernel_gridder_flops(int jobsize, int nr_timesteps, int nr_channels, int subgridsize, int nr_polarizations) {
    return
    1ULL * jobsize * nr_timesteps * subgridsize * subgridsize * nr_channels * (
        // Phasor
        2 * 22 +
        // UV
        nr_polarizations * 8) +
    // ATerm
    1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 30 +
    // Spheroidal
    1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 2 +
    // Shift
    1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 6;
}

uint64_t kernel_gridder_bytes(int jobsize, int nr_timesteps, int nr_channels, int subgridsize, int nr_polarizations) {
    return
    // Grid
    1ULL * jobsize * nr_timesteps * subgridsize * subgridsize * nr_channels * (nr_polarizations * sizeof(FLOAT_COMPLEX) + sizeof(float)) +
    // ATerm
    1ULL * jobsize * subgridsize * subgridsize * (2 * sizeof(unsigned)) + (2 * nr_polarizations * sizeof(FLOAT_COMPLEX) + sizeof(float)) +
    // Spheroidal
    1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * sizeof(FLOAT_COMPLEX);
}

}
#pragma omp end declare target
