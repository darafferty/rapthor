#include <cudawrappers/cu.hpp>

#include "KernelWTiling.h"

namespace idg::kernel::cuda {

/*
  Copy tiles
*/
std::string KernelWTilingCopy::source_file_ = "KernelWTiling_copy.cu";
std::string KernelWTilingCopy::kernel_function_ = "kernel_copy_tiles";

KernelWTilingCopy::KernelWTilingCopy(cu::Device& device, cu::Stream& stream,
                                     const cu::Module& module,
                                     const Parameters& parameters)
    : CompiledKernel(
          stream,
          std::make_unique<cu::Function>(module, kernel_function_.c_str()),
          parameters) {}

void KernelWTilingCopy::enqueue(
    unsigned int nr_polarizations, unsigned int nr_tiles,
    unsigned int src_tile_size, unsigned int dst_tile_size,
    cu::DeviceMemory& d_src_tile_ids, cu::DeviceMemory& d_dst_tile_ids,
    cu::DeviceMemory& d_src_tiles, cu::DeviceMemory& d_dst_tiles) {
  setArg(0, src_tile_size);
  setArg(1, dst_tile_size);
  setArg(2, d_src_tile_ids);
  setArg(3, d_dst_tile_ids);
  setArg(4, d_src_tiles);
  setArg(5, d_dst_tiles);

  Grid grid(nr_polarizations, nr_tiles);
  Block block(KernelWTilingPhasor::kBlockSizeX);

  setEnqueueWorkSizes(grid, block);

  launch();
}

template <>
CompileDefinitions KernelFactory<KernelWTilingCopy>::compileDefinitions()
    const {
  CompileDefinitions compile_definitions =
      KernelFactoryBase::compileDefinitions(parameters_);

  return compile_definitions;
}

/*
  Apply phasor
*/
std::string KernelWTilingPhasor::source_file_ = "KernelWTiling_phasor.cu";
std::string KernelWTilingPhasor::kernel_function_ = "kernel_apply_phasor";

KernelWTilingPhasor::KernelWTilingPhasor(cu::Device& device, cu::Stream& stream,
                                         const cu::Module& module,
                                         const Parameters& parameters)
    : CompiledKernel(
          stream,
          std::make_unique<cu::Function>(module, kernel_function_.c_str()),
          parameters) {}

void KernelWTilingPhasor::enqueue(
    unsigned int nr_polarizations, unsigned int nr_tiles, float image_size,
    float w_step, unsigned int tile_size, cu::DeviceMemory& d_tiles,
    cu::DeviceMemory& d_shift, cu::DeviceMemory& d_tile_coordinates, int sign) {
  setArg(0, image_size);
  setArg(1, w_step);
  setArg(2, tile_size);
  setArg(3, d_tiles);
  setArg(4, d_shift);
  setArg(5, d_tile_coordinates);
  setArg(6, sign);

  Grid grid(nr_polarizations, nr_tiles);
  Block block(KernelWTilingPhasor::kBlockSizeX);

  setEnqueueWorkSizes(grid, block);

  launch();
}

template <>
CompileDefinitions KernelFactory<KernelWTilingPhasor>::compileDefinitions()
    const {
  CompileDefinitions compile_definitions =
      KernelFactoryBase::compileDefinitions(parameters_);

  return compile_definitions;
}

/*
  Subgrids from wtiles
*/
std::string KernelWTilingSubgridsFromWtiles::source_file_ =
    "KernelWTiling_subgrids_from_wtiles.cu";
std::string KernelWTilingSubgridsFromWtiles::kernel_function_ =
    "kernel_subgrids_from_wtiles";

KernelWTilingSubgridsFromWtiles::KernelWTilingSubgridsFromWtiles(
    cu::Device& device, cu::Stream& stream, const cu::Module& module,
    const Parameters& parameters)
    : CompiledKernel(
          stream,
          std::make_unique<cu::Function>(module, kernel_function_.c_str()),
          parameters) {}

void KernelWTilingSubgridsFromWtiles::enqueue(
    int nr_subgrids, int nr_polarizations, long grid_size, int subgrid_size,
    int tile_size, int subgrid_offset, cu::DeviceMemory& d_metadata,
    cu::DeviceMemory& d_subgrid, cu::DeviceMemory& d_tiles) {
  setArg(0, nr_polarizations);
  setArg(1, grid_size);
  setArg(2, subgrid_size);
  setArg(3, tile_size);
  setArg(4, subgrid_offset);
  setArg(5, d_metadata);
  setArg(6, d_subgrid);
  setArg(7, d_tiles);

  Grid grid(nr_subgrids);
  Block block(KernelWTilingSubgridsFromWtiles::kBlockSizeX);

  setEnqueueWorkSizes(grid, block);

  launch();
}

template <>
CompileDefinitions
KernelFactory<KernelWTilingSubgridsFromWtiles>::compileDefinitions() const {
  CompileDefinitions compile_definitions =
      KernelFactoryBase::compileDefinitions(parameters_);

  return compile_definitions;
}

/*
  Subgrids to wtiles
*/
std::string KernelWTilingSubgridsToWtiles::source_file_ =
    "KernelWTiling_subgrids_to_wtiles.cu";
std::string KernelWTilingSubgridsToWtiles::kernel_function_ =
    "kernel_subgrids_to_wtiles";

KernelWTilingSubgridsToWtiles::KernelWTilingSubgridsToWtiles(
    cu::Device& device, cu::Stream& stream, const cu::Module& module,
    const Parameters& parameters)
    : CompiledKernel(
          stream,
          std::make_unique<cu::Function>(module, kernel_function_.c_str()),
          parameters) {}

void KernelWTilingSubgridsToWtiles::enqueue(
    int nr_subgrids, int nr_polarizations, long grid_size, int subgrid_size,
    int tile_size, int subgrid_offset, cu::DeviceMemory& d_metadata,
    cu::DeviceMemory& d_subgrid, cu::DeviceMemory& d_tiles,
    std::complex<float> scale) {
  setArg(0, nr_polarizations);
  setArg(1, grid_size);
  setArg(2, subgrid_size);
  setArg(3, tile_size);
  setArg(4, subgrid_offset);
  setArg(5, d_metadata);
  setArg(6, d_subgrid);
  setArg(7, d_tiles);
  setArg(8, scale);

  Grid grid(nr_subgrids);
  Block block(KernelWTilingSubgridsToWtiles::kBlockSizeX);

  setEnqueueWorkSizes(grid, block);

  launch();
}

template <>
CompileDefinitions
KernelFactory<KernelWTilingSubgridsToWtiles>::compileDefinitions() const {
  CompileDefinitions compile_definitions =
      KernelFactoryBase::compileDefinitions(parameters_);

  return compile_definitions;
}

/*
  Wtiles from patch
*/
std::string KernelWTilingWTilesFromPatch::source_file_ =
    "KernelWTiling_wtiles_from_patch.cu";
std::string KernelWTilingWTilesFromPatch::kernel_function_ =
    "kernel_wtiles_from_patch";

KernelWTilingWTilesFromPatch::KernelWTilingWTilesFromPatch(
    cu::Device& device, cu::Stream& stream, const cu::Module& module,
    const Parameters& parameters)
    : CompiledKernel(
          stream,
          std::make_unique<cu::Function>(module, kernel_function_.c_str()),
          parameters) {}

void KernelWTilingWTilesFromPatch::enqueue(
    int nr_polarizations, int nr_tiles, long grid_size, int tile_size,
    int padded_tile_size, int patch_size, idg::Coordinate patch_coordinate,
    cu::DeviceMemory& d_tile_ids, cu::DeviceMemory& d_tile_coordinates,
    cu::DeviceMemory& d_tiles, cu::DeviceMemory& d_patch) {
  setArg(0, nr_tiles);
  setArg(1, grid_size);
  setArg(2, tile_size);
  setArg(3, padded_tile_size);
  setArg(4, patch_size);
  setArg(5, patch_coordinate);
  setArg(6, d_tile_ids);
  setArg(7, d_tile_coordinates);
  setArg(8, d_tiles);
  setArg(9, d_patch);

  Grid grid(nr_polarizations, patch_size);
  Block block(KernelWTilingWTilesFromPatch::kBlockSizeX);

  setEnqueueWorkSizes(grid, block);

  launch();
}

template <>
CompileDefinitions
KernelFactory<KernelWTilingWTilesFromPatch>::compileDefinitions() const {
  CompileDefinitions compile_definitions =
      KernelFactoryBase::compileDefinitions(parameters_);

  return compile_definitions;
}

/*
  Wtiles to patch
*/
std::string KernelWTilingWtilesToPatch::source_file_ =
    "KernelWTiling_wtiles_to_patch.cu";
std::string KernelWTilingWtilesToPatch::kernel_function_ =
    "kernel_wtiles_to_patch";

KernelWTilingWtilesToPatch::KernelWTilingWtilesToPatch(
    cu::Device& device, cu::Stream& stream, const cu::Module& module,
    const Parameters& parameters)
    : CompiledKernel(
          stream,
          std::make_unique<cu::Function>(module, kernel_function_.c_str()),
          parameters) {}

void KernelWTilingWtilesToPatch::enqueue(
    int nr_polarizations, int nr_tiles, long grid_size, int tile_size,
    int padded_tile_size, int patch_size, idg::Coordinate patch_coordinate,
    cu::DeviceMemory& d_tile_ids, cu::DeviceMemory& d_tile_coordinates,
    cu::DeviceMemory& d_tiles, cu::DeviceMemory& d_patch) {
  setArg(0, nr_tiles);
  setArg(1, grid_size);
  setArg(2, tile_size);
  setArg(3, padded_tile_size);
  setArg(4, patch_size);
  setArg(5, patch_coordinate);
  setArg(6, d_tile_ids);
  setArg(7, d_tile_coordinates);
  setArg(8, d_tiles);
  setArg(9, d_patch);

  Grid grid(nr_polarizations, patch_size);
  Block block(KernelWTilingWtilesToPatch::kBlockSizeX);

  setEnqueueWorkSizes(grid, block);

  launch();
}

template <>
CompileDefinitions
KernelFactory<KernelWTilingWtilesToPatch>::compileDefinitions() const {
  CompileDefinitions compile_definitions =
      KernelFactoryBase::compileDefinitions(parameters_);

  return compile_definitions;
}

/*
  Wtiles to grid
*/
std::string KernelWTilingWtilesToGrid::source_file_ =
    "KernelWTiling_wtiles_to_grid.cu";
std::string KernelWTilingWtilesToGrid::kernel_function_ =
    "kernel_wtiles_to_grid";

KernelWTilingWtilesToGrid::KernelWTilingWtilesToGrid(
    cu::Device& device, cu::Stream& stream, const cu::Module& module,
    const Parameters& parameters)
    : CompiledKernel(
          stream,
          std::make_unique<cu::Function>(module, kernel_function_.c_str()),
          parameters) {}

void KernelWTilingWtilesToGrid::enqueue(int nr_polarizations, int nr_tiles,
                                        long grid_size, int tile_size,
                                        int padded_tile_size,
                                        cu::DeviceMemory& d_tile_ids,
                                        cu::DeviceMemory& d_tile_coordinates,
                                        cu::DeviceMemory& d_tiles,
                                        cu::DeviceMemory& d_grid) {
  setArg(0, grid_size);
  setArg(1, tile_size);
  setArg(2, padded_tile_size);
  setArg(3, d_tile_ids);
  setArg(4, d_tile_coordinates);
  setArg(5, d_tiles);
  setArg(6, d_grid);

  Grid grid(nr_polarizations, nr_tiles);
  Block block(KernelWTilingWtilesToGrid::kBlockSizeX);

  setEnqueueWorkSizes(grid, block);

  launch();
}

template <>
CompileDefinitions
KernelFactory<KernelWTilingWtilesToGrid>::compileDefinitions() const {
  CompileDefinitions compile_definitions =
      KernelFactoryBase::compileDefinitions(parameters_);

  return compile_definitions;
}

/*
  Wtiles from grid
*/
std::string KernelWTilingWtilesFromGrid::source_file_ =
    "KernelWTiling_wtiles_from_grid.cu";
std::string KernelWTilingWtilesFromGrid::kernel_function_ =
    "kernel_wtiles_from_grid";

KernelWTilingWtilesFromGrid::KernelWTilingWtilesFromGrid(
    cu::Device& device, cu::Stream& stream, const cu::Module& module,
    const Parameters& parameters)
    : CompiledKernel(
          stream,
          std::make_unique<cu::Function>(module, kernel_function_.c_str()),
          parameters) {}

void KernelWTilingWtilesFromGrid::enqueue(int nr_polarizations, int nr_tiles,
                                          long grid_size, int tile_size,
                                          int padded_tile_size,
                                          cu::DeviceMemory& d_tile_ids,
                                          cu::DeviceMemory& d_tile_coordinates,
                                          cu::DeviceMemory& d_tiles,
                                          cu::DeviceMemory& d_grid) {
  setArg(0, grid_size);
  setArg(1, tile_size);
  setArg(2, padded_tile_size);
  setArg(3, d_tile_ids);
  setArg(4, d_tile_coordinates);
  setArg(5, d_tiles);
  setArg(6, d_grid);

  Grid grid(nr_polarizations, nr_tiles);
  Block block(KernelWTilingWtilesFromGrid::kBlockSizeX);

  setEnqueueWorkSizes(grid, block);

  launch();
}

template <>
CompileDefinitions
KernelFactory<KernelWTilingWtilesFromGrid>::compileDefinitions() const {
  CompileDefinitions compile_definitions =
      KernelFactoryBase::compileDefinitions(parameters_);

  return compile_definitions;
}

}  // namespace idg::kernel::cuda
