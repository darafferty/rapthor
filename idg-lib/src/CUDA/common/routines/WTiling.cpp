#include "../CUDA.h"
#include "../InstanceCUDA.h"

namespace idg {
namespace proxy {
namespace cuda {

unsigned int CUDA::plan_tile_fft(unsigned int nr_polarizations,
                                 unsigned int nr_tiles_batch,
                                 const unsigned int w_padded_tile_size,
                                 const cu::Context& context,
                                 const size_t free_memory,
                                 std::unique_ptr<cufft::C2C_2D>& fft) const {
  // Determine the maximum batch size given the amount of
  // free device memory and the memory required for the FFT plan.
  size_t sizeof_w_padded_tile = w_padded_tile_size * w_padded_tile_size *
                                nr_polarizations * sizeof(std::complex<float>);
  unsigned int nr_tiles_batch_fft = (free_memory / sizeof_w_padded_tile) * 0.95;
  nr_tiles_batch = std::min(nr_tiles_batch, nr_tiles_batch_fft);

  // Make FFT plan
  unsigned batch = nr_tiles_batch * nr_polarizations;
  unsigned stride = 1;
  unsigned dist = w_padded_tile_size * w_padded_tile_size;
  while (!fft) {
    try {
      // Try to make a FFT plan
      fft.reset(new cufft::C2C_2D(context, w_padded_tile_size,
                                  w_padded_tile_size, stride, dist, batch));
    } catch (cufft::Error& e) {
      // Try again with a smaller batch size
      if (nr_tiles_batch > 1) {
        std::clog << __func__
                  << ": reducing nr_tiles_batch to: " << nr_tiles_batch
                  << std::endl;
        nr_tiles_batch *= 0.9;
        fft.reset();
      } else {
        std::cerr << __func__ << ": could not plan tile-fft." << std::endl;
        throw e;
      }
    }
  }

  // The new batch size
  return nr_tiles_batch;
}

size_t CUDA::bytes_required_wtiling(const WTileUpdateSet& wtile_set,
                                    const int nr_polarizations,
                                    const int subgrid_size,
                                    const float image_size, const float w_step,
                                    const idg::Array1D<float>& shift,
                                    const size_t bytes_free) const {
  if (wtile_set.empty()) {
    return 0;
  }

  // Compute the maximum padded w-tile size for any w-tile in the set
  int w_padded_tile_size =
      compute_w_padded_tile_size_max(wtile_set, m_tile_size, subgrid_size,
                                     image_size, w_step, shift(0), shift(1));

  // Compute the memory required for such a padded w-tile
  size_t sizeof_w_padded_tile = w_padded_tile_size * w_padded_tile_size *
                                nr_polarizations * sizeof(std::complex<float>);

  // Most of the w-tiling buffers are already allocated in ::init_cache,
  // but additional memory is needed for the FFT when batches of padded w-tiles
  // are processed. Compute the amount of required given that:
  //  - the memory required for the FFT is the same as the input size
  //  - at most half of the available device memory is used
  size_t nr_tiles_batch = (bytes_free / 2) / sizeof_w_padded_tile;
  size_t bytes_required = nr_tiles_batch * sizeof_w_padded_tile;

  // The batch size should be at least one
  if (nr_tiles_batch < 1) {
    std::stringstream message;
    message << "Could not reserve " << bytes_required
            << " for nr_tiles_batch =  " << nr_tiles_batch
            << ", with sizeof_w_padded_tile = " << sizeof_w_padded_tile;
    throw std::runtime_error(message.str());
  }

  return bytes_required;
}

void CUDA::init_buffers_wtiling(unsigned int subgrid_size) {
  const kernel::cuda::InstanceCUDA& device = get_device(0);
  const cu::Context& context = get_device(0).get_context();

  // Memory in use prior to allocating buffers for w-tiling
#if defined(DEBUG)
  size_t bytes_reserved = device.get_free_memory();
#endif

  // W-tiling has a three-level hierarchy:
  //  1) padded tiles (tile_size + subgrid_size)^2
  //  2) w-padded tiles (tile_size + subgrid_size + w_size)^2
  //  3) patches (patch_size)^2
  // When tiles need to be flushed, they are processed
  // in batches using a three-step process:
  //  1) subset of padded tiles -> subset of w-padded tiles
  //  2) subset of w-padded tiles -> patches
  //  3) patches -> grid
  // and vice versa when initializing tiles:
  //  1) grid -> patches
  //  2) patches -> w-padded tiles
  //  3) w-padded tiles -> padded tiles
  // While the padded tile size is known beforehand, the w-padded tile size
  // is data dependent. The code uses two pre-allocated buffers of
  // the same size:
  // d_tiles - for padded tiles
  // d_padded_tiles - for w-padded tiles.
  // This implies that the tiles are processed in batches.

  // Compute the size of one tile
  const int nr_polarizations = m_grid->get_z_dim();
  int tile_size = m_tile_size + subgrid_size;
  size_t sizeof_tile =
      nr_polarizations * tile_size * tile_size * sizeof(std::complex<float>);

  // Compute the number of tiles given that:
  //  - sizeof(d_tiles) == sizeof(d_padded_tiles)
  //  - at most 20% of the device memory is used for either one of these
  m_nr_tiles = (device.get_free_memory() * 0.2) / sizeof_tile;

  // Allocate the tile buffers
  size_t sizeof_tiles = m_nr_tiles * sizeof_tile;
  m_buffers_wtiling.d_tiles.reset(new cu::DeviceMemory(context, sizeof_tiles));
  m_buffers_wtiling.d_padded_tiles.reset(
      new cu::DeviceMemory(context, sizeof_tiles));
  m_buffers_wtiling.h_tiles.reset(new cu::HostMemory(context, sizeof_tiles));

  // An FFT plan for the padded tiles is allocated on-demand,
  // ::run_imaging calls bytes_required_wtiling to take the amount of
  // memory needed into account based on the current padded tile size.

  // Allocate patch buffers
  m_buffers_wtiling.d_patches.resize(m_nr_patches_batch);
  for (unsigned int i = 0; i < m_nr_patches_batch; i++) {
    size_t sizeof_patch = nr_polarizations * m_patch_size * m_patch_size *
                          sizeof(std::complex<float>);
    m_buffers_wtiling.d_patches[i].reset(
        new cu::DeviceMemory(context, sizeof_patch));
  }

// Report memory usage for w-tiling
#if defined(DEBUG)
  bytes_reserved -= device.get_free_memory();
  std::cout << "bytes_reserved = " << bytes_reserved
            << " (for w-tiling buffers)" << std::endl;
#endif
}

}  // end namespace cuda
}  // end namespace proxy
}  // end namespace idg