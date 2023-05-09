#include "math.cu"
#include "Types.h"

__device__ void update_pixel(
    const unsigned             nr_polarizations,
    const unsigned             subgrid_size,
    const unsigned             nr_stations,
    const unsigned             s,
    const unsigned             y,
    const unsigned             x,
    const unsigned             aterm_index,
    const unsigned             station1,
    const unsigned             station2,
    const float2* __restrict__ aterms,
          float2* __restrict__ pixel,
          float2* __restrict__ pixel_sum)
{
    // Load aterm for station1
    int station1_index = (aterm_index * nr_stations + station1) *
                              subgrid_size * subgrid_size * 4 +
                          y * subgrid_size * 4 +
                          x * 4;
    const float2 *aterm1 = &aterms[station1_index];

    // Load aterm for station2
    int station2_index = (aterm_index * nr_stations + station2) *
                              subgrid_size * subgrid_size * 4 +
                          y * subgrid_size * 4 +
                          x * 4;
    const float2 *aterm2 = &aterms[station2_index];

    // Apply aterm
    apply_aterm_gridder(pixel, aterm1, aterm2);

    // Update pixel
    if (nr_polarizations == 4) {
        // Full Stokes
        for (unsigned pol = 0; pol < nr_polarizations; pol++) {
            pixel_sum[pol] += pixel[pol];
        }
    } else if (nr_polarizations == 1) {
        // Stokes-I only
        pixel_sum[0] += pixel[0];
        pixel_sum[3] += pixel[3];
    }
}

extern "C" {
__global__ void kernel_gridder(
    const int                        time_offset,
    const int                        nr_polarizations,
    const int                        grid_size,
    const int                        subgrid_size,
    const float                      image_size,
    const float                      w_step,
    const float                      shift_l,
    const float                      shift_m,
    const int                        nr_channels,
    const int                        nr_stations,
    const UVW<float>*   __restrict__ uvw,
    const float*        __restrict__ wavenumbers,
    const float2*       __restrict__ visibilities,
    const float*        __restrict__ taper,
    const float2*       __restrict__ aterms,
    const unsigned int* __restrict__ aterm_indices,
    const Metadata*     __restrict__ metadata,
    const float2*       __restrict__ avg_aterm,
          float2*       __restrict__ subgrid)
{
  int s = blockIdx.x;
  int tid = threadIdx.x;
  int nr_threads = blockDim.x * blockDim.y;

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
    float2 pixel_cur[4];
    float2 pixel_sum[4];

    for (int j = 0; j < 4; j++) {
      pixel_cur[j] = make_float2(0, 0);
      pixel_sum[j] = make_float2(0, 0);
    }

    // Compute l,m,n
    float l = compute_l(x, subgrid_size, image_size);
    float m = compute_m(y, subgrid_size, image_size);
    float n = compute_n(l, m);

    // Initialize aterm index to first timestep
    unsigned int aterm_idx_previous = aterm_indices[time_offset_global];

    // Iterate all timesteps
    for (int time = 0; time < nr_timesteps; time++) {
      // Get aterm index for current timestep
      int time_current = time_offset_global + time;
      const unsigned int aterm_idx_current = aterm_indices[time_current];

      // Determine whether aterm has changed
      bool aterm_changed = aterm_idx_previous != aterm_idx_current;

      if (aterm_changed) {
        // Update pixel
        update_pixel(
            nr_polarizations, subgrid_size, nr_stations, s, y, x,
            aterm_idx_previous, station1, station2, aterms,
            pixel_cur, pixel_sum);

        // Reset pixel
        for (int pol = 0; pol < 4; pol++) {
            pixel_cur[pol] = make_float2(0, 0);
        }

        // Update aterm index
        aterm_idx_previous = aterm_idx_current;
      }

      // Load UVW coordinates
      float u = uvw[time_offset_global + time].u;
      float v = uvw[time_offset_global + time].v;
      float w = uvw[time_offset_global + time].w;

      // Compute phase index
      float phase_index = u * l + v * m + w * n;

      // Compute phase offset
      float phase_offset = u_offset * l + v_offset * m + w_offset * n;

      // Update pixel for every channel
      for (int chan = 0; chan < nr_channels; chan++) {
        // Compute phase
        float phase = phase_offset - (phase_index * wavenumbers[chan]);

        // Compute phasor
        float2 phasor = make_float2(cosf(phase), sinf(phase));

        // Update pixel for every polarization
        size_t index = (time_offset_global + time) * nr_channels + chan;
        if (nr_polarizations == 4) {
          pixel_cur[0] += visibilities[index * 4 + 0] * phasor;
          pixel_cur[1] += visibilities[index * 4 + 1] * phasor;
          pixel_cur[2] += visibilities[index * 4 + 2] * phasor;
          pixel_cur[3] += visibilities[index * 4 + 3] * phasor;
        } else if (nr_polarizations == 1) {
          pixel_cur[0] += visibilities[index * 2 + 0] * phasor;
          pixel_cur[3] += visibilities[index * 2 + 1] * phasor;
        }
      }
    } // end for time

    update_pixel(
        nr_polarizations, subgrid_size, nr_stations, s, y, x,
        aterm_idx_previous, station1, station2, aterms,
        pixel_cur, pixel_sum);

    // Load taper
    float sph = taper[y * subgrid_size + x];

    // Compute shifted position in subgrid
    int x_dst = (x + (subgrid_size/2)) % subgrid_size;
    int y_dst = (y + (subgrid_size/2)) % subgrid_size;

    // Set subgrid value
    if (nr_polarizations == 4) {
      for (int pol = 0; pol < nr_polarizations; pol++) {
        size_t index =
            s * nr_polarizations * subgrid_size * subgrid_size +
            pol * subgrid_size * subgrid_size + y_dst * subgrid_size + x_dst;
          subgrid[index] = pixel_sum[pol] * sph;
      }
    } else if (nr_polarizations == 1) {
        size_t index =
            s * subgrid_size * subgrid_size +
            y_dst * subgrid_size + x_dst;
        subgrid[index] = (pixel_sum[0] + pixel_sum[3]) * sph * 0.5;
    }
  } // end for i (pixels)
}
} // end extern "C"
