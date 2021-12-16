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

}  // end namespace cuda
}  // end namespace proxy
}  // end namespace idg