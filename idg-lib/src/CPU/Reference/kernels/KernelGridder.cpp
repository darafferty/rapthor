#include <complex>
#include <cmath>
#include <cstring>

#include "Types.h"
#include "Math.h"


extern "C" {
    void kernel_gridder(
        const int   nr_subgrids,
        const int   gridsize,
        const int   subgridsize,
        const float imagesize,
        const float w_step_in_lambda,
        const float* shift,
        const int   nr_channels,
        const int   nr_stations,
        const idg::UVWCoordinate<float>* uvw,
        const float*                     wavenumbers,
        const std::complex<float>*       visibilities,
        const float*                     spheroidal,
        const std::complex<float>*       aterms,
        const std::complex<float>*       avg_aterm_correction,
        const idg::Metadata*             metadata,
              std::complex<float>*       subgrid)
    {
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
                    const float l = compute_l(x, subgridsize, imagesize);
                    const float m = compute_m(y, subgridsize, imagesize);
                    const float n = compute_n(l, m, shift);

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

                    apply_aterm(
                        aXX1, aXY1, aYX1, aYY1,
                        aXX2, aXY2, aYX2, aYY2,
                        pixels);

                    if (avg_aterm_correction)
                    {
                        apply_avg_aterm_correction(avg_aterm_correction + (y*subgridsize + x)*16, pixels);
                    }

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
