#include "math.cu"
#include "Types.h"

extern "C" {
__global__ void kernel_gridder(
    const int                      time_offset,
    const int                      nr_polarizations,
    const int                      grid_size,
    const int                      subgrid_size,
    const float                    image_size,
    const float                    w_step,
    const float                    shift_l,
    const float                    shift_m,
    const int                      nr_channels,
    const int                      nr_stations,
    const UVW<float>* __restrict__ uvw,
    const float*      __restrict__ wavenumbers,
          float2*     __restrict__ visibilities,
    const float*      __restrict__ spheroidal,
    const float2*     __restrict__ aterms,
    const int*        __restrict__ aterms_indices,
    const Metadata*   __restrict__ metadata,
    const float2*     __restrict__ avg_aterm,
          float2*     __restrict__ subgrid)
{
  int s = blockIdx.x;
  int tid = threadIdx.x;
  int nr_threads = blockDim.x * blockDim.y;

  int nr_correlations = 0;
  if (nr_polarizations == 4) {
    nr_correlations = 4;
  } else if (nr_polarizations == 1) {
    nr_correlations = 2;
  }

  // Load metadata
  const Metadata &m = metadata[s];
  const int time_offset_global = m.time_index - time_offset;
  const int nr_timesteps = m.nr_timesteps;
  const int x_coordinate = m.coordinate.x;
  const int y_coordinate = m.coordinate.y;
  const int station1 = m.baseline.station1;
  const int station2 = m.baseline.station2;

  // Compute u,v,w offset in wavelenghts
  const float u_offset = (x_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
  const float v_offset = (y_coordinate + subgrid_size/2 - grid_size/2) / image_size * 2 * M_PI;
  const float w_offset = w_step * ((float) m.coordinate.z + 0.5) * 2 * M_PI;

  // Iterate all pixels in subgrid
  for (int i = tid; i < subgrid_size * subgrid_size; i += nr_threads) {

    int y = i / subgrid_size;
    int x = i % subgrid_size;
    if (y >= subgrid_size) {
      break;
    }

    // Initialize pixel for every polarization
    float2 pixel_sum[4];
    float2 pixel_cur[4];

    for (int j = 0; j < nr_correlations; j++) {
      pixel_sum[j] = make_float2(0, 0);
    }

    // Compute l,m,n
    float l = compute_l(x, subgrid_size, image_size);
    float m = compute_m(y, subgrid_size, image_size);
    float n = compute_n(l, m);

    // Iterate all timesteps
    for (int time = 0; time < nr_timesteps; time++) {
      // Load UVW coordinates
      float u = uvw[time_offset_global + time].u;
      float v = uvw[time_offset_global + time].v;
      float w = uvw[time_offset_global + time].w;

      // Compute phase index
      float phase_index = u * l + v * m + w * n;

      // Compute phase offset
      float phase_offset = u_offset * l + v_offset * m + w_offset * n;

      for (int j = 0; j < nr_correlations; j++) {
        pixel_cur[j] = make_float2(0, 0);
      }

      // Update pixel for every channel
      for (int chan = 0; chan < nr_channels; chan++) {
        // Compute phase
        float phase = phase_offset - (phase_index * wavenumbers[chan]);

        // Compute phasor
        float2 phasor = make_float2(cos(phase), sin(phase));

        // Update pixel for every polarization
        size_t index = (time_offset_global + time) * nr_channels + chan;
        for (int pol = 0; pol < nr_correlations; pol++) {
          float2 visibility = visibilities[index * nr_correlations + pol];
          pixel_cur[pol] += visibility * phasor;
        }
      }

      int aterm_index = aterms_indices[time_offset_global + time];

      // Load a term for station1
      int station1_index = (aterm_index * nr_stations + station1) *
                                subgrid_size * subgrid_size * nr_correlations +
                            y * subgrid_size * nr_correlations +
                            x * nr_correlations;
      const float2 *aterm1_ptr = &aterms[station1_index];

      // Load aterm for station2
      int station2_index = (aterm_index * nr_stations + station2) *
                                subgrid_size * subgrid_size * nr_correlations +
                            y * subgrid_size * nr_correlations +
                            x * nr_correlations;
      const float2 *aterm2_ptr = &aterms[station2_index];

      // Apply aterm
      apply_aterm_gridder(pixel_cur, aterm1_ptr, aterm2_ptr);

      if (nr_polarizations == 4) {
          // Full Stokes
          for (unsigned pol = 0; pol < nr_polarizations; pol++) {
              pixel_sum[pol] += pixel_cur[pol];
          }
      } else if (nr_polarizations == 1) {
          // Stokes-I only
          pixel_sum[0] += pixel_cur[0];
          pixel_sum[3] += pixel_cur[3];
      }
    }

    // Load spheroidal
    float sph = spheroidal[y * subgrid_size + x];

    // Compute shifted position in subgrid
    int x_dst = (x + (subgrid_size/2)) % subgrid_size;
    int y_dst = (y + (subgrid_size/2)) % subgrid_size;

    // Set subgrid value
    for (int pol = 0; pol < nr_correlations; pol++) {
      unsigned idx_subgrid =
          s * nr_correlations * subgrid_size * subgrid_size +
          pol * subgrid_size * subgrid_size + y_dst * subgrid_size + x_dst;
      subgrid[idx_subgrid] = pixel_sum[pol] * sph;
    }
  }
}
} // end extern "C"
