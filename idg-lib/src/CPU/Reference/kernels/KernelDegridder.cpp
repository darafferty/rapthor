#include <complex>
#include <cmath>
#include <cstring>

#include "Types.h"
#include "Math.h"


extern "C" {
    void kernel_degridder(
        const int   nr_subgrids,
        const int   gridsize,
        const int   subgridsize,
        const float imagesize,
        const float w_step_in_lambda,
        const float* __restrict__ shift,
        const int   nr_channels,
        const int   nr_stations,
        const idg::UVWCoordinate<float>* uvw,
        const float*                     wavenumbers,
              std::complex<float>*       visibilities,
        const float*                     spheroidal,
        const std::complex<float>*       aterms,
        const idg::Metadata*             metadata,
        const std::complex<float>*       subgrid)
    {
        // Find offset of first subgrid
        const idg::Metadata m       = metadata[0];
        const int baseline_offset_1 = m.baseline_offset;

        // Iterate all subgrids
        #pragma omp parallel for
        for (int s = 0; s < nr_subgrids; s++) {

            // Load metadata
            const idg::Metadata m  = metadata[s];
            const int local_offset = (m.baseline_offset - baseline_offset_1) + m.time_offset;
            const int nr_timesteps = m.nr_timesteps;
            const int aterm_index  = m.aterm_index;
            const int station1     = m.baseline.station1;
            const int station2     = m.baseline.station2;
            const int x_coordinate = m.coordinate.x;
            const int y_coordinate = m.coordinate.y;
            const float w_offset_in_lambda = w_step_in_lambda * (m.coordinate.z + 0.5);

            // Storage
            std::complex<float> pixels[subgridsize][subgridsize][NR_POLARIZATIONS];

            // Apply aterm to subgrid
            for (int y = 0; y < subgridsize; y++) {
                for (int x = 0; x < subgridsize; x++) {
                    // Load aterm for station1
                    int station1_index =
                        (aterm_index * nr_stations + station1) *
                        subgridsize * subgridsize * NR_POLARIZATIONS +
                        y * subgridsize * NR_POLARIZATIONS + x * NR_POLARIZATIONS;
                    std::complex<float> aXX1 = conj(aterms[station1_index + 0]);
                    std::complex<float> aXY1 = conj(aterms[station1_index + 1]);
                    std::complex<float> aYX1 = conj(aterms[station1_index + 2]);
                    std::complex<float> aYY1 = conj(aterms[station1_index + 3]);

                    // Load aterm for station2
                    int station2_index =
                        (aterm_index * nr_stations + station2) *
                        subgridsize * subgridsize * NR_POLARIZATIONS +
                        y * subgridsize * NR_POLARIZATIONS + x * NR_POLARIZATIONS;
                    std::complex<float> aXX2 = aterms[station2_index + 0];
                    std::complex<float> aXY2 = aterms[station2_index + 1];
                    std::complex<float> aYX2 = aterms[station2_index + 2];
                    std::complex<float> aYY2 = aterms[station2_index + 3];

                    // Load spheroidal
                    float sph = spheroidal[y * subgridsize + x];

                    // Compute shifted position in subgrid
                    int x_src = (x + (subgridsize/2)) % subgridsize;
                    int y_src = (y + (subgridsize/2)) % subgridsize;

                    // Load uv values
                    std::complex<float> pixels_[NR_POLARIZATIONS];
                    pixels_[0] = sph * subgrid[
                        s * NR_POLARIZATIONS * subgridsize * subgridsize +
                        0 * subgridsize * subgridsize + y_src * subgridsize + x_src];
                    pixels_[1] = sph * subgrid[
                        s * NR_POLARIZATIONS * subgridsize * subgridsize +
                        1 * subgridsize * subgridsize + y_src * subgridsize + x_src];
                    pixels_[2] = sph * subgrid[
                        s * NR_POLARIZATIONS * subgridsize * subgridsize +
                        2 * subgridsize * subgridsize + y_src * subgridsize + x_src];
                    pixels_[3] = sph * subgrid[
                        s * NR_POLARIZATIONS * subgridsize * subgridsize +
                        3 * subgridsize * subgridsize + y_src * subgridsize + x_src];

                    apply_aterm(
                        aXX1, aXY1, aYX1, aYY1,
                        aXX2, aXY2, aYX2, aYY2,
                        pixels_);

                    for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        pixels[y][x][pol] = pixels_[pol];
                    }
#if 0
                    // Apply aterm to subgrid: P*A1^H
                    // [ pixels[0], pixels[1];    [ conj(aXX1), conj(aYX1);
                    //   pixels[2], pixels[3] ] *   conj(aXY1), conj(aYY1) ]
                    pixels[y][x][0]  = pixelsXX * aXX1;
                    pixels[y][x][0] += pixelsXY * aXY1;
                    pixels[y][x][1]  = pixelsXX * aYX1;
                    pixels[y][x][1] += pixelsXY * aYY1;
                    pixels[y][x][2]  = pixelsYX * aXX1;
                    pixels[y][x][2] += pixelsYY * aXY1;
                    pixels[y][x][3]  = pixelsYX * aYX1;
                    pixels[y][x][3] += pixelsYY * aYY1;

                    // Apply aterm to subgrid: A2*P
                    // [ aXX2, aXY1;      [ pixels[0], pixels[1];
                    //   aYX1, aYY2 ]  *    pixels[2], pixels[3] ]
                    pixelsXX = pixels[y][x][0];
                    pixelsXY = pixels[y][x][1];
                    pixelsYX = pixels[y][x][2];
                    pixelsYY = pixels[y][x][3];
                    pixels[y][x][0]  = pixelsXX * aXX2;
                    pixels[y][x][0] += pixelsYX * aXY2;
                    pixels[y][x][1]  = pixelsXY * aXX2;
                    pixels[y][x][1] += pixelsYY * aXY2;
                    pixels[y][x][2]  = pixelsXX * aYX2;
                    pixels[y][x][2] += pixelsYX * aYY2;
                    pixels[y][x][3]  = pixelsXY * aYX2;
                    pixels[y][x][3] += pixelsYY * aYY2;
#endif
                } // end x
            } // end y

            // Compute u and v offset in wavelenghts
            const float u_offset = (x_coordinate + subgridsize/2 - gridsize/2)
                                   * (2*M_PI / imagesize);
            const float v_offset = (y_coordinate + subgridsize/2 - gridsize/2)
                                   * (2*M_PI / imagesize);
            const float w_offset = 2*M_PI * w_offset_in_lambda;

            // Iterate all timesteps
            for (int time = 0; time < nr_timesteps; time++) {
                // Load UVW coordinates
                float u = uvw[local_offset + time].u;
                float v = uvw[local_offset + time].v;
                float w = uvw[local_offset + time].w;

                // Iterate all channels
                for (int chan = 0; chan < nr_channels; chan++) {

                    // Update all polarizations
                    std::complex<float> sum[NR_POLARIZATIONS];
                    memset(sum, 0, NR_POLARIZATIONS * sizeof(std::complex<float>));

                    // Iterate all pixels in subgrid
                    for (int y = 0; y < subgridsize; y++) {
                        for (int x = 0; x < subgridsize; x++) {

                            // Compute l,m,n
                            const float l = compute_l(x, subgridsize, imagesize);
                            const float m = compute_m(y, subgridsize, imagesize);
                            const float n = compute_n(l, m, shift);

                            // Compute phase index
                            float phase_index = u*l + v*m + w*n;

                            // Compute phase offset
                            float phase_offset = u_offset*l + v_offset*m + w_offset*n;

                            // Compute phase
                            float phase = (phase_index * wavenumbers[chan]) - phase_offset;

                            // Compute phasor
                            std::complex<float> phasor = {cosf(phase), sinf(phase)};

                            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                                sum[pol] += pixels[y][x][pol] * phasor;
                            }
                        } // end for x
                    } // end for y

                    const float scale = 1.0f / (subgridsize*subgridsize);
                    size_t index = (local_offset + time)*nr_channels + chan;
                    for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        visibilities[index * NR_POLARIZATIONS + pol] = sum[pol] * scale;
                    }
                } // end for channel
            } // end for time
        } // end for s
    } // end kernel_degridder
} // end extern "C"
