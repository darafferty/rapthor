#include <complex>
#include <cmath>
#include <cstring>

#include "Types.h"


extern "C" {
    void kernel_gridder(
        const int   nr_subgrids,
        const int   gridsize,
        const int   subgridsize,
        const float imagesize,
        const float w_offset_in_lambda,
        const int   nr_channels,
        const int   nr_stations,
        const idg::UVWCoordinate<float>* uvw,
        const float*                     wavenumbers,
        const std::complex<float>*       visibilities,
        const float*                     spheroidal,
        const std::complex<float>*       aterms,
        const idg::Metadata*             metadata,
              std::complex<float>*       subgrid)
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

            // Compute u and v offset in wavelenghts
            const float u_offset = (x_coordinate + subgridsize/2 - gridsize/2)
                                   * (2*M_PI / imagesize);
            const float v_offset = (y_coordinate + subgridsize/2 - gridsize/2)
                                   * (2*M_PI / imagesize);
            const float w_offset = 2*M_PI * w_offset_in_lambda;

            // Iterate all pixels in subgrid
            for (int y = 0; y < subgridsize; y++) {
                for (int x = 0; x < subgridsize; x++) {
                    // Initialize pixel for every polarization
                    std::complex<float> pixels[NR_POLARIZATIONS];
                    memset(pixels, 0, NR_POLARIZATIONS * sizeof(std::complex<float>));

                    // Compute l,m,n
                    const float l = (x+0.5-(subgridsize/2)) * imagesize/subgridsize;
                    const float m = (y+0.5-(subgridsize/2)) * imagesize/subgridsize;
                    // evaluate n = 1.0f - sqrt(1.0 - (l * l) - (m * m));
                    // accurately for small values of l and m
                    const float tmp = (l * l) + (m * m);
                    const float n = tmp / (1.0f + sqrtf(1.0f - tmp));

                    // Iterate all timesteps
                    for (int time = 0; time < nr_timesteps; time++) {
                        // Load UVW coordinates
                        float u = uvw[offset + time].u;
                        float v = uvw[offset + time].v;
                        float w = uvw[offset + time].w;

                        // Compute phase index
                        float phase_index = u*l + v*m + w*n;

                        // Compute phase offset
                        float phase_offset = u_offset*l + v_offset*m + w_offset*n;

                        // Update pixel for every channel
                        for (int chan = 0; chan < nr_channels; chan++) {
                            // Compute phase
                            float phase = phase_offset - (phase_index * wavenumbers[chan]);

                            // Compute phasor
                            std::complex<float> phasor = {cosf(phase), sinf(phase)};

                            // Update pixel for every polarization
                            size_t index = (offset + time)*nr_channels + chan;
                            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                                std::complex<float> visibility = visibilities[index * NR_POLARIZATIONS + pol];
                                pixels[pol] += visibility * phasor;
                            }
                        }
                    }

                    // Load a term for station1
                    int station1_index =
                        (aterm_index * nr_stations + station1) *
                        subgridsize * subgridsize * NR_POLARIZATIONS +
                        y * subgridsize * NR_POLARIZATIONS + x * NR_POLARIZATIONS;
                    std::complex<float> aXX1 = aterms[station1_index + 0];
                    std::complex<float> aXY1 = aterms[station1_index + 1];
                    std::complex<float> aYX1 = aterms[station1_index + 2];
                    std::complex<float> aYY1 = aterms[station1_index + 3];

                    // Load aterm for station2
                    int station2_index =
                        (aterm_index * nr_stations + station2) *
                        subgridsize * subgridsize * NR_POLARIZATIONS +
                        y * subgridsize * NR_POLARIZATIONS + x * NR_POLARIZATIONS;
                    std::complex<float> aXX2 = conj(aterms[station2_index + 0]);
                    std::complex<float> aXY2 = conj(aterms[station2_index + 1]);
                    std::complex<float> aYX2 = conj(aterms[station2_index + 2]);
                    std::complex<float> aYY2 = conj(aterms[station2_index + 3]);

                    // Apply aterm to subgrid: P*A1
                    // [ pixels[0], pixels[1];    [ aXX1, aXY1;
                    //   pixels[2], pixels[3] ] *   aYX1, aYY1 ]
                    std::complex<float> pixelsXX = pixels[0];
                    std::complex<float> pixelsXY = pixels[1];
                    std::complex<float> pixelsYX = pixels[2];
                    std::complex<float> pixelsYY = pixels[3];
                    pixels[0]  = (pixelsXX * aXX1);
                    pixels[0] += (pixelsXY * aYX1);
                    pixels[1]  = (pixelsXX * aXY1);
                    pixels[1] += (pixelsXY * aYY1);
                    pixels[2]  = (pixelsYX * aXX1);
                    pixels[2] += (pixelsYY * aYX1);
                    pixels[3]  = (pixelsYX * aXY1);
                    pixels[3] += (pixelsYY * aYY1);

                    // Apply aterm to subgrid: A2^H*P
                    // [ aXX2, aYX1;      [ pixels[0], pixels[1];
                    //   aXY1, aYY2 ]  *    pixels[2], pixels[3] ]
                    pixelsXX = pixels[0];
                    pixelsXY = pixels[1];
                    pixelsYX = pixels[2];
                    pixelsYY = pixels[3];
                    pixels[0]  = (pixelsXX * aXX2);
                    pixels[0] += (pixelsYX * aYX2);
                    pixels[1]  = (pixelsXY * aXX2);
                    pixels[1] += (pixelsYY * aYX2);
                    pixels[2]  = (pixelsXX * aXY2);
                    pixels[2] += (pixelsYX * aYY2);
                    pixels[3]  = (pixelsXY * aXY2);
                    pixels[3] += (pixelsYY * aYY2);

                    // Load spheroidal
                    float sph = spheroidal[y * subgridsize + x];

                    // Compute shifted position in subgrid
                    int x_dst = (x + (subgridsize/2)) % subgridsize;
                    int y_dst = (y + (subgridsize/2)) % subgridsize;

                    // Set subgrid value
                    for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        subgrid[s * NR_POLARIZATIONS * subgridsize * subgridsize +
                                pol * subgridsize * subgridsize + y_dst * subgridsize +
                                x_dst] = pixels[pol] * sph;
                    }
                } // end x
            } // end y
        } // end s
    }  // end kernel_gridder
}  // end extern C
