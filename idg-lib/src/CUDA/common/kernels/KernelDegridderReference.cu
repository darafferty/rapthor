#include "math.cu"
#include "Types.h"

extern "C" {
__global__ void kernel_degridder(
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
          float2*       __restrict__ visibilities,
    const float*        __restrict__ taper,
    const float2*       __restrict__ aterms,
    const unsigned int* __restrict__ aterm_indices,
    const Metadata*     __restrict__ metadata,
    const float2*       __restrict__ subgrid)
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

  // Iterate all visibilities
  for (int i = tid; i < nr_timesteps * nr_channels; i += nr_threads) {
    int time = i / nr_channels;
    int chan = i % nr_channels;

    if (time >= nr_timesteps) {
      break;
    }

    // Initialize visibility to zero
    float2 visibility[4];
    for (int j = 0; j < 4; j++) {
      visibility[j] = make_float2(0, 0);
    }

    // Load UVW coordinates
    float u = uvw[time_offset_global + time].u;
    float v = uvw[time_offset_global + time].v;
    float w = uvw[time_offset_global + time].w;

    // Iterate all pixels in subgrid
    for (int j = 0; j < subgrid_size * subgrid_size; j++) {
      int y = j / subgrid_size;
      int x = j % subgrid_size;

      if (y >= subgrid_size) {
        break;
      }

      // Load taper
      float sph = taper[y * subgrid_size + x];

      // Compute shifted position in subgrid
      int x_src = (x + (subgrid_size / 2)) % subgrid_size;
      int y_src = (y + (subgrid_size / 2)) % subgrid_size;

      // Load pixel value and apply taper
      float2 pixel[4];
      if (nr_polarizations == 4) {
        for (int pol = 0; pol < nr_polarizations; pol++) {
          size_t index =
              s * nr_polarizations * subgrid_size * subgrid_size +
              pol * subgrid_size * subgrid_size + y_src * subgrid_size +
              x_src;
          pixel[pol] = sph * subgrid[index];
        }
      } else if (nr_polarizations == 1) {
        size_t index = s * nr_polarizations * subgrid_size * subgrid_size +
                        y_src * subgrid_size + x_src;
        pixel[0] = sph * subgrid[index];
        pixel[1] = make_float2(0, 0);
        pixel[2] = make_float2(0, 0);
        pixel[3] = sph * subgrid[index];
      }

      const unsigned int aterm_index = aterm_indices[time_offset_global + time];

      // Load a term for station1
      int station1_index = (aterm_index * nr_stations + station1) *
                                subgrid_size * subgrid_size * 4 +
                            y * subgrid_size * 4 + x * 4;
      const float2 *aterm1_ptr = &aterms[station1_index];

      // Load aterm for station2
      int station2_index = (aterm_index * nr_stations + station2) *
                                subgrid_size * subgrid_size * 4 +
                            y * subgrid_size * 4 + x * 4;
      const float2 *aterm2_ptr = &aterms[station2_index];

      // Apply aterm
      apply_aterm_degridder(pixel, aterm1_ptr, aterm2_ptr);

      // Compute l,m,n
      float l = compute_l(x, subgrid_size, image_size);
      float m = compute_m(y, subgrid_size, image_size);
      float n = compute_n(l, m);

      // Compute phase index
      float phase_index = u * l + v * m + w * n;

      // Compute phase offset
      float phase_offset = u_offset * l + v_offset * m + w_offset * n;

      // Compute phase
      float phase = (phase_index * wavenumbers[chan]) - phase_offset;

      // Compute phasor
      float2 phasor = make_float2(cosf(phase), sinf(phase));

      // Update visibility
      if (nr_polarizations == 4) {
        for (int pol = 0; pol < 4; pol++) {
          visibility[pol] += pixel[pol] * phasor;
        }
      } else if (nr_polarizations == 1) {
        visibility[0] += pixel[0] * phasor;
        visibility[1] += pixel[3] * phasor;
      }
    } // end for j (pixels)

    // Store visibility
    const float scale = 1.0f / (subgrid_size * subgrid_size);
    if (nr_polarizations == 4) {
      size_t index = (time_offset_global + time) * nr_channels + chan;
      visibilities[index * 4 + 0] = visibility[0] * scale;
      visibilities[index * 4 + 1] = visibility[1] * scale;
      visibilities[index * 4 + 2] = visibility[2] * scale;
      visibilities[index * 4 + 3] = visibility[3] * scale;
    } else if (nr_polarizations == 1) {
      size_t index = (time_offset_global + time) * nr_channels + chan;
      visibilities[index * 2 + 0] = visibility[0] * scale;
      visibilities[index * 2 + 1] = visibility[3] * scale;
    }
  } // end for i (visibilities)
}

} // end extern "C"
