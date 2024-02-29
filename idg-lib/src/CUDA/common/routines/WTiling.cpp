#include <algorithm>

#include <cudawrappers/cu.hpp>
#include <cudawrappers/nvtx.hpp>

#include "../CUDA.h"
#include "../InstanceCUDA.h"
#include "../kernels/KernelFFT.h"

#include "common/WTiling.h"

namespace {
size_t compute_sizeof_padded_tile(size_t tile_size, size_t subgrid_size,
                                  size_t nr_polarizations) {
  // Compute the size of one tile
  const size_t padded_tile_size = tile_size + subgrid_size;
  const size_t sizeof_padded_tile = nr_polarizations * padded_tile_size *
                                    padded_tile_size *
                                    sizeof(std::complex<float>);

  return sizeof_padded_tile;
}

size_t compute_sizeof_patch(size_t patch_size, size_t nr_polarizations) {
  const size_t sizeof_patch =
      nr_polarizations * patch_size * patch_size * sizeof(std::complex<float>);
  return sizeof_patch;
}

}  // namespace

namespace idg {
namespace proxy {
namespace cuda {

size_t CUDA::bytes_required_wtiling(const WTileUpdateSet& wtile_set,
                                    const int nr_polarizations,
                                    const int subgrid_size,
                                    const float image_size, const float w_step,
                                    const std::array<float, 2>& shift,
                                    const size_t bytes_free) const {
  if (wtile_set.empty()) {
    return 0;
  }

  // Compute the maximum padded w-tile size for any w-tile in the set
  int w_padded_tile_size = compute_w_padded_tile_size_max(
      wtile_set, m_tile_size, subgrid_size, image_size, w_step, shift);

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
  const kernel::cuda::InstanceCUDA& device = get_device();

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
  const size_t nr_polarizations = get_grid().shape(1);
  const size_t tile_size = m_tile_size + subgrid_size;
  const size_t sizeof_tile =
      nr_polarizations * tile_size * tile_size * sizeof(std::complex<float>);

  // Compute the number of tiles given that:
  //  - sizeof(d_tiles) == sizeof(d_padded_tiles)
  //  - at most 20% of the device memory is used for either one of these
  m_nr_tiles = (device.get_free_memory() * 0.2) / sizeof_tile;

  // Allocate the tile buffers
  const size_t sizeof_tiles = m_nr_tiles * sizeof_tile;
  m_buffers_wtiling.d_tiles.reset(
      new cu::DeviceMemory(sizeof_tiles, CU_MEMORYTYPE_DEVICE));
  m_buffers_wtiling.d_padded_tiles.reset(
      new cu::DeviceMemory(sizeof_tiles, CU_MEMORYTYPE_DEVICE));
  m_buffers_wtiling.h_tiles.reset(new cu::HostMemory(sizeof_tiles));

  // An FFT plan for the padded tiles is allocated on-demand,
  // ::run_imaging calls bytes_required_wtiling to take the amount of
  // memory needed into account based on the current padded tile size.

  // Allocate patch buffers
  m_buffers_wtiling.d_patches.resize(m_nr_patches_batch);
  for (unsigned int i = 0; i < m_nr_patches_batch; i++) {
    const size_t sizeof_patch = nr_polarizations * m_patch_size * m_patch_size *
                                sizeof(std::complex<float>);
    m_buffers_wtiling.d_patches[i].reset(
        new cu::DeviceMemory(sizeof_patch, CU_MEMORYTYPE_DEVICE));
  }

// Report memory usage for w-tiling
#if defined(DEBUG)
  bytes_reserved -= device.get_free_memory();
  std::cout << "bytes_reserved = " << bytes_reserved
            << " (for w-tiling buffers)" << std::endl;
#endif
}

void CUDA::run_wtiles_to_grid(unsigned int subgrid_size, float image_size,
                              float w_step, const std::array<float, 2>& shift,
                              WTileUpdateInfo& wtile_flush_info) {
  // Load grid parameters
  const size_t nr_polarizations = get_grid().shape(1);
  const size_t grid_size = get_grid().shape(2);
  assert(get_grid().shape(3) == grid_size);

  // Load CUDA objects
  kernel::cuda::InstanceCUDA& device = get_device();
  cu::Stream& executestream = device.get_execute_stream();
  cu::Stream& dtohstream = device.get_dtoh_stream();

  // Load buffers
  cu::DeviceMemory& d_tiles = *m_buffers_wtiling.d_tiles;
  cu::DeviceMemory& d_padded_tiles = *m_buffers_wtiling.d_padded_tiles;
  cu::HostMemory& h_padded_tiles = *m_buffers_wtiling.h_tiles;

  // Get information on what wtiles to flush
  const int tile_size = m_tile_size;
  const int padded_tile_size = tile_size + subgrid_size;
  const size_t nr_tiles = wtile_flush_info.wtile_ids.size();
  std::vector<idg::Coordinate>& tile_coordinates =
      wtile_flush_info.wtile_coordinates;
  std::vector<int>& tile_ids = wtile_flush_info.wtile_ids;

  // Sort wtile_flush_info
  sort_by_patches(grid_size, tile_size, padded_tile_size, m_patch_size,
                  nr_tiles, wtile_flush_info);

  // Compute w_padded_tile_size for all tiles
  const float image_size_shift =
      image_size + 2 * std::max(std::abs(shift[0]), std::abs(shift[1]));
  std::vector<int> w_padded_tile_sizes = compute_w_padded_tile_sizes(
      tile_coordinates.data(), nr_tiles, w_step, image_size, image_size_shift,
      padded_tile_size);

  // Find the maximum tile size for all padded tiles
  int w_padded_tile_size =
      *std::max_element(w_padded_tile_sizes.begin(), w_padded_tile_sizes.end());

  // Compute the number of padded tiles
  const size_t sizeof_padded_tile =
      compute_sizeof_padded_tile(m_tile_size, subgrid_size, nr_polarizations);
  const size_t sizeof_padded_tiles = sizeof_padded_tile * m_nr_tiles;
  size_t sizeof_w_padded_tile = w_padded_tile_size * w_padded_tile_size *
                                nr_polarizations * sizeof(std::complex<float>);
  size_t nr_tiles_batch = (sizeof_padded_tiles / sizeof_w_padded_tile) / 2;
  nr_tiles_batch = std::min(nr_tiles_batch, nr_tiles);

  // Allocate coordinates buffer
  size_t sizeof_tile_coordinates = nr_tiles_batch * sizeof(idg::Coordinate);
  cu::DeviceMemory d_tile_coordinates(sizeof_tile_coordinates,
                                      CU_MEMORYTYPE_DEVICE);

  // Allocate ids buffer
  size_t sizeof_tile_ids = nr_tiles_batch * sizeof(int);
  cu::DeviceMemory d_tile_ids(sizeof_tile_ids, CU_MEMORYTYPE_DEVICE);
  cu::DeviceMemory d_padded_tile_ids(sizeof_tile_ids, CU_MEMORYTYPE_DEVICE);
  cu::DeviceMemory d_packed_tile_ids(sizeof_tile_ids, CU_MEMORYTYPE_DEVICE);

  // Initialize d_padded_tile_ids
  std::vector<int> padded_tile_ids(nr_tiles_batch);
  for (unsigned int i = 0; i < nr_tiles_batch; i++) {
    padded_tile_ids[i] = i;
  }

  // Copy shift to device
  const size_t sizeof_shift = shift.size() * sizeof(float);
  cu::DeviceMemory d_shift(sizeof_shift, CU_MEMORYTYPE_DEVICE);
  executestream.memcpyHtoDAsync(d_shift, shift.data(), sizeof_shift);

  // FFT kernel
  std::unique_ptr<kernel::cuda::KernelFFT> fft_kernel;

  // Iterate all tiles
  unsigned int last_w_padded_tile_size = w_padded_tile_size;
  unsigned int current_nr_tiles = nr_tiles_batch;
  for (unsigned int tile_offset = 0; tile_offset < nr_tiles;
       tile_offset += current_nr_tiles) {
    current_nr_tiles = std::min(nr_tiles_batch, nr_tiles - tile_offset);

    // Set w_padded_tile_size for current batch of tiles
    unsigned int current_w_padded_tile_size = *std::max_element(
        w_padded_tile_sizes.begin() + tile_offset,
        w_padded_tile_sizes.begin() + tile_offset + current_nr_tiles);

    if (tile_offset == 0 ||
        current_w_padded_tile_size != last_w_padded_tile_size) {
      // Initialize FFT for w_padded_tiles
      fft_kernel = device.plan_batched_fft(current_w_padded_tile_size,
                                           current_nr_tiles * nr_polarizations);
      current_nr_tiles = fft_kernel->batch() / nr_polarizations;
      last_w_padded_tile_size = current_w_padded_tile_size;
    }

    // Copy tile metadata to GPU
    sizeof_tile_ids = current_nr_tiles * sizeof(int);
    executestream.memcpyHtoDAsync(d_tile_ids, &tile_ids[tile_offset],
                                  sizeof_tile_ids);
    executestream.memcpyHtoDAsync(d_padded_tile_ids, padded_tile_ids.data(),
                                  sizeof_tile_ids);
    sizeof_tile_coordinates = current_nr_tiles * sizeof(idg::Coordinate);
    executestream.memcpyHtoDAsync(d_tile_coordinates,
                                  &tile_coordinates[tile_offset],
                                  sizeof_tile_coordinates);

    // Call kernel_copy_tiles
    device.launch_copy_tiles(nr_polarizations, current_nr_tiles,
                             padded_tile_size, current_w_padded_tile_size,
                             d_tile_ids, d_padded_tile_ids, d_tiles,
                             d_padded_tiles);

    // Launch inverse FFT
    device.launch_batched_fft(*fft_kernel, d_padded_tiles,
                              current_nr_tiles * nr_polarizations,
                              FourierDomainToImageDomain);

    // Call kernel_apply_phasor
    device.launch_apply_phasor_to_wtiles(
        nr_polarizations, current_nr_tiles, image_size, w_step,
        current_w_padded_tile_size, d_padded_tiles, d_shift, d_tile_coordinates,
        -1);

    // Launch forward FFT
    device.launch_batched_fft(*fft_kernel, d_padded_tiles,
                              current_nr_tiles * nr_polarizations,
                              ImageDomainToFourierDomain);

    // Wait for GPU to finish
    executestream.synchronize();

    // Find all tiles that (partially) fit in the current patch
    std::vector<idg::Coordinate> patch_coordinates;
    std::vector<int> patch_nr_tiles;
    std::vector<int> patch_tile_ids;
    std::vector<int> patch_tile_id_offsets;
    find_patches_for_tiles(
        grid_size, tile_size, current_w_padded_tile_size, m_patch_size,
        current_nr_tiles, &tile_coordinates[tile_offset], patch_coordinates,
        patch_nr_tiles, patch_tile_ids, patch_tile_id_offsets);
    const size_t total_nr_patches = patch_coordinates.size();

    // Iterate patches in batches (note: reusing h_padded_tiles for patches)
    const size_t sizeof_patch = nr_polarizations * m_patch_size * m_patch_size *
                                sizeof(std::complex<float>);
    const size_t max_nr_patches = sizeof_padded_tiles / sizeof_patch;
    size_t current_nr_patches = max_nr_patches;

    // Events
    std::vector<cu::Event> gpuFinished;
    std::vector<cu::Event> outputCopied;
    gpuFinished.reserve(m_nr_patches_batch);
    outputCopied.reserve(m_nr_patches_batch);
    for (unsigned int i = 0; i < m_nr_patches_batch; i++) {
      gpuFinished.emplace_back();
      outputCopied.emplace_back();
    }

    for (size_t patch_offset = 0; patch_offset < total_nr_patches;
         patch_offset += current_nr_patches) {
      current_nr_patches =
          std::min(current_nr_patches, total_nr_patches - patch_offset);

      for (unsigned int i = 0; i < current_nr_patches; i++) {
        int id = i % m_nr_patches_batch;
        cu::DeviceMemory& d_patch = *(m_buffers_wtiling.d_patches[id]);

        // Wait for previous patch to be computed
        if (i > m_nr_patches_batch) {
          executestream.wait(outputCopied[id]);
        }

        // Get patch metadata
        int patch_id = patch_offset + i;
        int* packed_tile_ids = &patch_tile_ids[patch_tile_id_offsets[patch_id]];
        idg::Coordinate patch_coordinate = patch_coordinates[patch_id];

        // Copy packed tile ids to GPU
        executestream.memcpyHtoDAsync(d_packed_tile_ids, packed_tile_ids,
                                      patch_nr_tiles[patch_id] * sizeof(int));

        // Reset patch
        executestream.wait(outputCopied[id]);
        executestream.zero(d_patch, sizeof_patch);

        // Combine tiles onto patch
        device.launch_adder_wtiles_to_patch(
            nr_polarizations, patch_nr_tiles[patch_id], grid_size,
            padded_tile_size - subgrid_size, current_w_padded_tile_size,
            m_patch_size, patch_coordinate, d_packed_tile_ids,
            d_tile_coordinates, d_padded_tiles, d_patch);
        executestream.record(gpuFinished[id]);

        // Copy patch to the host
        void* patch_ptr = static_cast<char*>(h_padded_tiles) + i * sizeof_patch;
        dtohstream.wait(gpuFinished[id]);
        dtohstream.memcpyDtoHAsync(patch_ptr, d_patch, sizeof_patch);
        dtohstream.record(outputCopied[id]);
      }

      // Wait for patches to be copied
      dtohstream.synchronize();

      // Add patch to the grid
      nvtx::Marker marker("patch_to_grid", nvtx::Marker::red);
      marker.start();

      run_adder_patch_to_grid(
          nr_polarizations, grid_size, m_patch_size, current_nr_patches,
          &patch_coordinates[patch_offset], get_grid().data(),
          static_cast<std::complex<float>*>(h_padded_tiles));
      marker.end();
    }  // end for patch_offset
  }    // end for tile_offset
}

void CUDA::run_subgrids_to_wtiles(
    unsigned nr_polarizations, unsigned int subgrid_offset,
    unsigned int nr_subgrids, unsigned int subgrid_size, float image_size,
    float w_step, const std::array<float, 2>& shift,
    WTileUpdateSet& wtile_flush_set, cu::DeviceMemory& d_subgrids,
    cu::DeviceMemory& d_metadata) {
  // Load CUDA objects
  kernel::cuda::InstanceCUDA& device = get_device();
  cu::Stream& stream = device.get_execute_stream();
  cu::DeviceMemory& d_tiles = *m_buffers_wtiling.d_tiles;

  // Performance measurement
  pmt::State startState;
  pmt::State endState;
  startState = device.measure();

  for (unsigned int subgrid_index = 0; subgrid_index < nr_subgrids;) {
    // Is a flush needed right now?
    if (!wtile_flush_set.empty() && wtile_flush_set.front().subgrid_index ==
                                        int(subgrid_index + subgrid_offset)) {
      // Get information on what wtiles to flush
      WTileUpdateInfo& wtile_flush_info = wtile_flush_set.front();

      // Project wtiles to master grid
      run_wtiles_to_grid(subgrid_size, image_size, w_step, shift,
                         wtile_flush_info);

      // Remove the flush event from the queue
      wtile_flush_set.pop_front();
    }

    // Initialize number of subgrids to process next to all remaining subgrids
    // in job
    int nr_subgrids_to_process = nr_subgrids - subgrid_index;

    // Check whether a flush needs to happen before the end of the job
    if (!wtile_flush_set.empty() &&
        wtile_flush_set.front().subgrid_index -
                int(subgrid_index + subgrid_offset) <
            nr_subgrids_to_process) {
      // Reduce the number of subgrids to process to just before the next flush
      // event
      nr_subgrids_to_process = wtile_flush_set.front().subgrid_index -
                               (subgrid_index + subgrid_offset);
    }

    // Add all subgrids to the wtiles
    const size_t grid_size = get_grid().shape(2);
    assert(get_grid().shape(3) == grid_size);
    const size_t N = subgrid_size * subgrid_size;
    const std::complex<float> scale(1.0f / N, 1.0f / N);
    device.launch_adder_subgrids_to_wtiles(
        nr_subgrids_to_process, nr_polarizations, grid_size, subgrid_size,
        m_tile_size, subgrid_index, d_metadata, d_subgrids, d_tiles, scale);
    stream.synchronize();

    // Increment the subgrid index by the actual number of processed subgrids
    subgrid_index += nr_subgrids_to_process;
  }

  // End performance measurement
  endState = device.measure();
  get_report()->update(Report::wtiling_forward, startState, endState);
}

void CUDA::run_wtiles_from_grid(unsigned int subgrid_size, float image_size,
                                float w_step, const std::array<float, 2>& shift,
                                WTileUpdateInfo& wtile_initialize_info) {
  // Load grid parameters
  const size_t nr_polarizations = get_grid().shape(1);
  const size_t grid_size = get_grid().shape(2);
  assert(get_grid().shape(3) == grid_size);

  // Load CUDA objects
  kernel::cuda::InstanceCUDA& device = get_device();
  cu::Stream& executestream = device.get_execute_stream();
  cu::Stream& htodstream = device.get_htod_stream();

  // Load buffers
  cu::DeviceMemory& d_tiles = *m_buffers_wtiling.d_tiles;
  cu::DeviceMemory& d_padded_tiles = *m_buffers_wtiling.d_padded_tiles;
  cu::HostMemory& h_padded_tiles = *m_buffers_wtiling.h_tiles;

  // Get information on what wtiles to flush
  const int tile_size = m_tile_size;
  const int padded_tile_size = tile_size + subgrid_size;
  const size_t nr_tiles = wtile_initialize_info.wtile_ids.size();
  std::vector<idg::Coordinate>& tile_coordinates =
      wtile_initialize_info.wtile_coordinates;
  std::vector<int>& tile_ids = wtile_initialize_info.wtile_ids;

  // Sort wtile_initialize_info
  sort_by_patches(grid_size, tile_size, padded_tile_size, m_patch_size,
                  nr_tiles, wtile_initialize_info);

  // Compute w_padded_tile_size for all tiles
  const float image_size_shift =
      image_size + 2 * std::max(std::abs(shift[0]), std::abs(shift[1]));
  std::vector<int> w_padded_tile_sizes = compute_w_padded_tile_sizes(
      tile_coordinates.data(), nr_tiles, w_step, image_size, image_size_shift,
      padded_tile_size);

  // Find the maximum tile size for all padded tiles
  int w_padded_tile_size =
      *std::max_element(w_padded_tile_sizes.begin(), w_padded_tile_sizes.end());

  // Compute the number of padded tiles
  const size_t sizeof_w_padded_tile = w_padded_tile_size * w_padded_tile_size *
                                      nr_polarizations *
                                      sizeof(std::complex<float>);
  size_t nr_tiles_batch = (d_padded_tiles.size() / sizeof_w_padded_tile) / 2;
  nr_tiles_batch = std::min(nr_tiles_batch, nr_tiles);

  // Allocate coordinates buffer
  unsigned int sizeof_tile_coordinates =
      nr_tiles_batch * sizeof(idg::Coordinate);
  cu::DeviceMemory d_tile_coordinates(sizeof_tile_coordinates,
                                      CU_MEMORYTYPE_DEVICE);

  // Allocate ids buffer
  unsigned int sizeof_tile_ids = nr_tiles_batch * sizeof(int);
  cu::DeviceMemory d_tile_ids(sizeof_tile_ids, CU_MEMORYTYPE_DEVICE);
  cu::DeviceMemory d_padded_tile_ids(sizeof_tile_ids, CU_MEMORYTYPE_DEVICE);
  cu::DeviceMemory d_packed_tile_ids(sizeof_tile_ids, CU_MEMORYTYPE_DEVICE);

  // Initialize d_padded_tile_ids
  std::vector<int> padded_tile_ids(nr_tiles_batch);
  for (unsigned int i = 0; i < nr_tiles_batch; i++) {
    padded_tile_ids[i] = i;
  }
  executestream.memcpyHtoDAsync(d_padded_tile_ids, padded_tile_ids.data(),
                                sizeof_tile_ids);

  // Copy shift to device
  const size_t sizeof_shift = shift.size() * sizeof(float);
  cu::DeviceMemory d_shift(sizeof_shift, CU_MEMORYTYPE_DEVICE);
  executestream.memcpyHtoDAsync(d_shift, shift.data(), sizeof_shift);

  // FFT kernel
  std::unique_ptr<kernel::cuda::KernelFFT> fft_kernel;

  // Iterate all tiles
  unsigned int last_w_padded_tile_size = w_padded_tile_size;
  unsigned int current_nr_tiles = nr_tiles_batch;
  for (unsigned int tile_offset = 0; tile_offset < nr_tiles;
       tile_offset += current_nr_tiles) {
    current_nr_tiles = std::min(nr_tiles_batch, nr_tiles - tile_offset);

    // Set w_padded_tile_size for current job
    unsigned int current_w_padded_tile_size = *std::max_element(
        w_padded_tile_sizes.begin() + tile_offset,
        w_padded_tile_sizes.begin() + tile_offset + current_nr_tiles);

    if (tile_offset == 0 ||
        current_w_padded_tile_size != last_w_padded_tile_size) {
      // Initialize FFT for w_padded_tiles
      fft_kernel = device.plan_batched_fft(current_w_padded_tile_size,
                                           current_nr_tiles * nr_polarizations);
      current_nr_tiles = fft_kernel->batch() / nr_polarizations;
      last_w_padded_tile_size = current_w_padded_tile_size;
    }

    // Copy tile metadata to GPU
    sizeof_tile_ids = current_nr_tiles * sizeof(int);
    executestream.memcpyHtoDAsync(d_tile_ids, &tile_ids[tile_offset],
                                  sizeof_tile_ids);
    executestream.memcpyHtoDAsync(d_padded_tile_ids, padded_tile_ids.data(),
                                  sizeof_tile_ids);
    sizeof_tile_coordinates = current_nr_tiles * sizeof(idg::Coordinate);
    executestream.memcpyHtoDAsync(d_tile_coordinates,
                                  &tile_coordinates[tile_offset],
                                  sizeof_tile_coordinates);
    // Find all tiles that (partially) fit in the current patch
    std::vector<idg::Coordinate> patch_coordinates;
    std::vector<int> patch_nr_tiles;
    std::vector<int> patch_tile_ids;
    std::vector<int> patch_tile_id_offsets;
    find_patches_for_tiles(
        grid_size, tile_size, current_w_padded_tile_size, m_patch_size,
        current_nr_tiles, &tile_coordinates[tile_offset], patch_coordinates,
        patch_nr_tiles, patch_tile_ids, patch_tile_id_offsets);
    const size_t total_nr_patches = patch_coordinates.size();

    // Iterate patches in batches (note: reusing h_padded_tiles for patches)
    const size_t sizeof_patch =
        compute_sizeof_patch(m_patch_size, nr_polarizations);
    const size_t sizeof_padded_tile =
        compute_sizeof_padded_tile(m_tile_size, subgrid_size, nr_polarizations);
    const size_t sizeof_padded_tiles = sizeof_padded_tile * m_nr_tiles;
    const size_t max_nr_patches = sizeof_padded_tiles / sizeof_patch;
    size_t current_nr_patches = max_nr_patches;

    // Events
    std::vector<cu::Event> inputCopied;
    std::vector<cu::Event> gpuFinished;
    inputCopied.reserve(m_nr_patches_batch);
    gpuFinished.reserve(m_nr_patches_batch);
    for (unsigned int i = 0; i < m_nr_patches_batch; i++) {
      inputCopied.emplace_back();
      gpuFinished.emplace_back();
    }

    // Reset padded tiles
    executestream.zero(d_padded_tiles, sizeof_padded_tiles);

    for (size_t patch_offset = 0; patch_offset < total_nr_patches;
         patch_offset += current_nr_patches) {
      current_nr_patches =
          std::min(current_nr_patches, total_nr_patches - patch_offset);

      // Split patch from grid
      nvtx::Marker marker("patch_from_grid", nvtx::Marker::red);
      marker.start();
      run_splitter_patch_from_grid(
          nr_polarizations, grid_size, m_patch_size, current_nr_patches,
          &patch_coordinates[patch_offset], get_grid().data(),
          static_cast<std::complex<float>*>(h_padded_tiles));
      marker.end();

      for (unsigned int i = 0; i < current_nr_patches; i++) {
        int id = i % m_nr_patches_batch;
        cu::DeviceMemory& d_patch = *(m_buffers_wtiling.d_patches[id]);

        // Wait for previous patch to be computed
        if (i > m_nr_patches_batch) {
          gpuFinished[id].synchronize();
        }

        // Get patch metadata
        int patch_id = patch_offset + i;
        int* packed_tile_ids = &patch_tile_ids[patch_tile_id_offsets[patch_id]];
        idg::Coordinate patch_coordinate = patch_coordinates[patch_id];

        // Copy packed tile ids to GPU
        sizeof_tile_ids = current_nr_tiles * sizeof(int);
        executestream.memcpyHtoDAsync(d_packed_tile_ids, packed_tile_ids,
                                      patch_nr_tiles[patch_id] * sizeof(int));

        // Copy patch to the GPU
        void* patch_ptr = static_cast<char*>(h_padded_tiles) + i * sizeof_patch;
        htodstream.wait(gpuFinished[id]);
        htodstream.memcpyHtoDAsync(d_patch, patch_ptr, sizeof_patch);
        htodstream.record(inputCopied[id]);

        // Read tile from patch
        executestream.wait(inputCopied[id]);
        device.launch_splitter_wtiles_from_patch(
            nr_polarizations, patch_nr_tiles[patch_id], grid_size,
            padded_tile_size - subgrid_size, current_w_padded_tile_size,
            m_patch_size, patch_coordinate, d_packed_tile_ids,
            d_tile_coordinates, d_padded_tiles, d_patch);
        executestream.record(gpuFinished[id]);
      }

      // Wait for tiles to be created
      executestream.synchronize();

    }  // end for patch_offset

    // Launch inverse FFT
    device.launch_batched_fft(*fft_kernel, d_padded_tiles,
                              current_nr_tiles * nr_polarizations,
                              FourierDomainToImageDomain);

    // Call kernel_apply_phasor
    device.launch_apply_phasor_to_wtiles(
        nr_polarizations, current_nr_tiles, image_size, w_step,
        current_w_padded_tile_size, d_padded_tiles, d_shift, d_tile_coordinates,
        1);

    // Launch forward FFT
    device.launch_batched_fft(*fft_kernel, d_padded_tiles,
                              current_nr_tiles * nr_polarizations,
                              ImageDomainToFourierDomain);

    // Call kernel_copy_tiles
    device.launch_copy_tiles(nr_polarizations, current_nr_tiles,
                             current_w_padded_tile_size, padded_tile_size,
                             d_padded_tile_ids, d_tile_ids, d_padded_tiles,
                             d_tiles);

    // Wait for tiles to be copied
    executestream.synchronize();
  }  // end for tile_offset
}

void CUDA::run_subgrids_from_wtiles(
    unsigned int nr_polarizations, unsigned int subgrid_offset,
    unsigned int nr_subgrids, unsigned int subgrid_size, float image_size,
    float w_step, const std::array<float, 2>& shift,
    WTileUpdateSet& wtile_initialize_set, cu::DeviceMemory& d_subgrids,
    cu::DeviceMemory& d_metadata) {
  // Load CUDA objects
  kernel::cuda::InstanceCUDA& device = get_device();
  cu::Stream& stream = device.get_execute_stream();

  // Load buffers
  cu::DeviceMemory& d_tiles = *m_buffers_wtiling.d_tiles;

  // Performance measurement
  pmt::State startState;
  pmt::State endState;
  startState = device.measure();

  for (unsigned int subgrid_index = 0; subgrid_index < nr_subgrids;) {
    // Check whether initialize is needed right now
    if (!wtile_initialize_set.empty() &&
        wtile_initialize_set.front().subgrid_index ==
            (int)(subgrid_index + subgrid_offset)) {
      // Get information on what wtiles to initialize
      WTileUpdateInfo& wtile_initialize_info = wtile_initialize_set.front();

      // Initialize the wtiles from the grid
      run_wtiles_from_grid(subgrid_size, image_size, w_step, shift,
                           wtile_initialize_info);

      // Remove the initialize event from the queue
      wtile_initialize_set.pop_front();
    }

    // Initialize number of subgrids to process next to all remaining subgrids
    // in job
    int nr_subgrids_to_process = nr_subgrids - subgrid_index;

    // Check whether initialization needs to happen before the end of the job
    if (!wtile_initialize_set.empty() &&
        wtile_initialize_set.front().subgrid_index -
                (int)(subgrid_index + subgrid_offset) <
            nr_subgrids_to_process) {
      // Reduce the number of subgrids to process to just before the next
      // initialization event
      nr_subgrids_to_process = wtile_initialize_set.front().subgrid_index -
                               (subgrid_offset + subgrid_index);
    }

    // Process all subgrids that can be processed now
    const size_t grid_size = get_grid().shape(2);
    assert(get_grid().shape(3) == grid_size);
    device.launch_splitter_subgrids_from_wtiles(
        nr_subgrids_to_process, nr_polarizations, grid_size, subgrid_size,
        m_tile_size, subgrid_index, d_metadata, d_subgrids, d_tiles);
    stream.synchronize();

    // Increment the subgrid index by the actual number of processed subgrids
    subgrid_index += nr_subgrids_to_process;
  }

  // End performance measurement
  endState = device.measure();
  get_report()->update(Report::wtiling_backward, startState, endState);
}

void CUDA::flush_wtiles() {
  // Get parameters
  const size_t nr_polarizations = get_grid().shape(1);
  const size_t grid_size = get_grid().shape(2);
  assert(get_grid().shape(3) == grid_size);
  float cell_size = m_cache_state.cell_size;
  float image_size = grid_size * cell_size;
  int subgrid_size = m_cache_state.subgrid_size;
  float w_step = m_cache_state.w_step;
  const std::array<float, 2>& shift = m_cache_state.shift;

  // Get all the remaining wtiles
  WTileUpdateInfo wtile_flush_info = m_wtiles.clear();

  // Project wtiles to master grid
  if (wtile_flush_info.wtile_ids.size()) {
    get_report()->initialize();
    kernel::cuda::InstanceCUDA& device = get_device();
    pmt::State startState;
    pmt::State endState;
    startState = device.measure();
    run_wtiles_to_grid(subgrid_size, image_size, w_step, shift,
                       wtile_flush_info);
    endState = device.measure();
    get_report()->update(Report::wtiling_forward, startState, endState);
    get_report()->print_total(nr_polarizations);
  }
}

}  // end namespace cuda
}  // end namespace proxy
}  // end namespace idg