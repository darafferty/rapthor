#include "Types.h"
#include "math.cu"

#include "KernelCalibrate_index.cuh"

extern "C" {

__global__ void kernel_calibrate_hessian(
    const int                        nr_polarizations,
    const int                        total_nr_timesteps,
    const int                        nr_channels,
    const int                        term_offset_y,
    const int                        term_offset_x,
    const int                        nr_terms,
    const float*        __restrict__ weights,
    const unsigned int* __restrict__ aterm_indices,
    const Metadata*     __restrict__ metadata,
    const float2*       __restrict__ sums_y,
    const float2*       __restrict__ sums_x,
          double*       __restrict__ hessian)
{
    unsigned tidx       = threadIdx.x;
    unsigned tidy       = threadIdx.y;
    unsigned s          = blockIdx.x;

    // Metadata for current subgrid
    const Metadata &m = metadata[s];
    const unsigned int time_offset_global = m.time_index;
    const unsigned int nr_timesteps       = m.nr_timesteps;
    const unsigned int channel_begin      = m.channel_begin;
    const unsigned int channel_end        = m.channel_end;
    const unsigned int nr_channels_local  = channel_end - channel_begin;

    // Iterate timesteps
    int current_nr_timesteps = 0;
    for (int time_offset_local = 0; time_offset_local < nr_timesteps; time_offset_local += current_nr_timesteps) {
        const unsigned int aterm_idx = aterm_indices[time_offset_global + time_offset_local];

        // Determine number of timesteps to process
        current_nr_timesteps = 0;
        for (int time = time_offset_local; time < nr_timesteps; time++) {
            if (aterm_indices[time_offset_global + time] == aterm_idx) {
                current_nr_timesteps++;
            } else {
                break;
            }
        }

        // Set term nubmers
        unsigned int term_nr1 = tidx;
        unsigned int term_nr0 = tidy;

        // Compute hessian update
        double update = 0.0;

        // Iterate all timesteps
        for (unsigned int time = 0; time < current_nr_timesteps; time++) {

            // Iterate all channels
            for (unsigned int chan = 0; chan < nr_channels_local; chan++) {

                // Iterate all polarizations
                for (unsigned int pol = 0; pol < nr_polarizations; pol++) {
                    unsigned int time_idx_global = time_offset_global + time_offset_local + time;
                    unsigned int chan_idx_local  = channel_begin + chan;
                    unsigned int  vis_idx = index_visibility(4, nr_channels, time_idx_global, chan_idx_local, pol);
                    unsigned int sum_idx0 = index_sums(4, total_nr_timesteps, nr_channels, term_nr0, pol, time_idx_global, chan_idx_local);
                    unsigned int sum_idx1 = index_sums(4, total_nr_timesteps, nr_channels, term_nr1, pol, time_idx_global, chan_idx_local);
                    float2 sum0 = sums_y[sum_idx0];
                    float2 sum1 = sums_x[sum_idx1] * weights[vis_idx];

                    // Update hessian
                    update += sum0.x * sum1.x + sum0.y * sum1.y;
                } // end for pol
            } // end chan
        } // end for time

        // Compute term indices
        unsigned int term_idx1 = term_offset_x + term_nr1;
        unsigned int term_idx0 = term_offset_y + term_nr0;

        // Update hessian
        unsigned int idx = aterm_idx * nr_terms * nr_terms +
                           term_idx1 * nr_terms + term_idx0;
        atomicAdd(&hessian[idx], update);

        // Update mirror hessian
        if (term_offset_y != term_offset_x) {
            unsigned int idx = aterm_idx * nr_terms * nr_terms +
                               term_idx0 * nr_terms + term_idx1;
            atomicAdd(&hessian[idx], update);
        }
    } // end for time_offset_local
} // end kernel_calibrate_hessian

} // end extern "C"
