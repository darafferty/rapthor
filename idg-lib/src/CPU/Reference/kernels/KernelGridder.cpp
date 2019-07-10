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
        const idg::UVW<float>*           uvw,
        const float*                     wavenumbers,
        const std::complex<float>*       visibilities,
        const float*                     spheroidal,
        const std::complex<float>*       aterms,
        const int*                       aterms_indices,
        const std::complex<float>*       avg_aterm_correction,
        const idg::Metadata*             metadata,
              std::complex<float>*       subgrid)
    {
        // Iterate all subgrids
        #pragma omp parallel for
        for (int s = 0; s < nr_subgrids; s++) {
            // Load metadata
            const idg::Metadata m   = metadata[s];
            const int time_offset   = m.time_index;
            const int nr_timesteps  = m.nr_timesteps;
            const int channel_begin = m.channel_begin;
            const int channel_end   = m.channel_end;
            const int station1      = m.baseline.station1;
            const int station2      = m.baseline.station2;
            const int x_coordinate  = m.coordinate.x;
            const int y_coordinate  = m.coordinate.y;
            const float w_offset_in_lambda = w_step_in_lambda * (m.coordinate.z + 0.5);

            // Compute u and v offset in wavelenghts
            const float u_offset = (x_coordinate + subgridsize/2 - gridsize/2)
                                   * (2*M_PI / imagesize);
            const float v_offset = (y_coordinate + subgridsize/2 - gridsize/2)
                                   * (2*M_PI / imagesize);
            const float w_offset = 2*M_PI * w_offset_in_lambda;

            // Storage
            std::complex<float> pixels[NR_POLARIZATIONS][subgridsize][subgridsize];
            memset(pixels, 0, subgridsize*subgridsize*NR_POLARIZATIONS*sizeof(std::complex<float>));

            // Iterate all pixels in subgrid
            for (int y = 0; y < subgridsize; y++) {
                for (int x = 0; x < subgridsize; x++) {

                    // Compute l,m,n
                    const float l = compute_l(x, subgridsize, imagesize);
                    const float m = compute_m(y, subgridsize, imagesize);
                    const float n = compute_n(l, m, shift);


                    // Iterate all timesteps
                    for (int time = 0; time < nr_timesteps; time++) {
                        // Pixel
                        std::complex<float> pixel[NR_POLARIZATIONS];
                        memset(pixel, 0, NR_POLARIZATIONS * sizeof(std::complex<float>));

                        // Load UVW coordinates
                        float u = uvw[time_offset + time].u;
                        float v = uvw[time_offset + time].v;
                        float w = uvw[time_offset + time].w;

                        // Compute phase index
                        float phase_index = u*l + v*m + w*n;

                        // Compute phase offset
                        float phase_offset = u_offset*l + v_offset*m + w_offset*n;

                        // Update pixel for every channel
                        for (int chan = channel_begin; chan < channel_end; chan++) {
                            // Compute phase
                            float phase = phase_offset - (phase_index * wavenumbers[chan]);

                            // Compute phasor
                            std::complex<float> phasor = {cosf(phase), sinf(phase)};

                            // Update pixel for every polarization
                            size_t index = (time_offset + time)*nr_channels + chan;
                            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                                std::complex<float> visibility = visibilities[index * NR_POLARIZATIONS + pol];
                                pixel[pol] += visibility * phasor;
                            }
                        } // end for channel

                        // Load aterm index
                        int aterm_index = aterms_indices[time_offset + time];

                        // Load a term for station1
                        int station1_index =
                            (aterm_index * nr_stations + station1) *
                            subgridsize * subgridsize * NR_POLARIZATIONS +
                            y * subgridsize * NR_POLARIZATIONS + x * NR_POLARIZATIONS;
                        const std::complex<float> *aterms1 = (std::complex<float> *) &aterms[station1_index];

                        // Load aterm for station2
                        int station2_index =
                            (aterm_index * nr_stations + station2) *
                            subgridsize * subgridsize * NR_POLARIZATIONS +
                            y * subgridsize * NR_POLARIZATIONS + x * NR_POLARIZATIONS;
                        const std::complex<float> *aterms2 = (std::complex<float> *) &aterms[station2_index];

                        apply_aterm_gridder(pixel, aterms1, aterms2);

                        // Update pixel
                        for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                            pixels[pol][y][x] += pixel[pol];
                        }
                    } // end for time

                    // Load pixel
                    std::complex<float> pixel[NR_POLARIZATIONS];
                    for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        pixel[pol] = pixels[pol][y][x];
                    }

                    if (avg_aterm_correction)
                    {
                        apply_avg_aterm_correction(avg_aterm_correction + (y*subgridsize + x)*16, pixel);
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
                                x_dst] = pixel[pol] * sph;
                    }
                } // end for x
            } // end for y
        } // end for s
    }  // end kernel_gridder
}  // end extern C
