#ifndef IDG_CUDA_KERNEL_WTILING_H_
#define IDG_CUDA_KERNEL_WTILING_H_

#include <complex>
#include <string>

#include "../KernelFactory.h"
#include "Kernel.h"

namespace idg {
#include "common/KernelTypes.h"
}

namespace idg::kernel::cuda {

/*
  Copy tiles
*/
class KernelWTilingCopy : public CompiledKernel {
 public:
  static std::string source_file_;
  static std::string kernel_function_;

  KernelWTilingCopy(cu::Device& device, cu::Stream& stream,
                    const cu::Module& module,
                    const Parameters& parameters = {});

  void enqueue(unsigned int nr_polarizations, unsigned int nr_tiles,
               unsigned int src_tile_size, unsigned int dst_tile_size,
               cu::DeviceMemory& d_src_tile_ids,
               cu::DeviceMemory& d_dst_tile_ids, cu::DeviceMemory& d_src_tiles,
               cu::DeviceMemory& d_dst_tiles);

  static constexpr unsigned kBlockSizeX = 128;
};

template <>
CompileDefinitions KernelFactory<KernelWTilingCopy>::compileDefinitions() const;

/*
  Apply phasor
*/
class KernelWTilingPhasor : public CompiledKernel {
 public:
  static std::string source_file_;
  static std::string kernel_function_;

  KernelWTilingPhasor(cu::Device& device, cu::Stream& stream,
                      const cu::Module& module,
                      const Parameters& parameters = {});

  void enqueue(unsigned int nr_polarizations, unsigned int nr_tiles,
               float image_size, float w_step, unsigned int tile_size,
               cu::DeviceMemory& d_tiles, cu::DeviceMemory& d_shift,
               cu::DeviceMemory& d_tile_coordinates, int sign);

  static constexpr unsigned kBlockSizeX = 128;
};

template <>
CompileDefinitions KernelFactory<KernelWTilingPhasor>::compileDefinitions()
    const;

/*
  Subgrids from wtiles
*/
class KernelWTilingSubgridsFromWtiles : public CompiledKernel {
 public:
  static std::string source_file_;
  static std::string kernel_function_;

  KernelWTilingSubgridsFromWtiles(cu::Device& device, cu::Stream& stream,
                                  const cu::Module& module,
                                  const Parameters& parameters = {});

  void enqueue(int nr_subgrids, int nr_polarizations, long grid_size,
               int subgrid_size, int tile_size, int subgrid_offset,
               cu::DeviceMemory& d_metadata, cu::DeviceMemory& d_subgrid,
               cu::DeviceMemory& d_tiles);

  static constexpr unsigned kBlockSizeX = 128;
};

template <>
CompileDefinitions
KernelFactory<KernelWTilingSubgridsFromWtiles>::compileDefinitions() const;

/*
  Subgrids to wtiles
*/
class KernelWTilingSubgridsToWtiles : public CompiledKernel {
 public:
  static std::string source_file_;
  static std::string kernel_function_;

  KernelWTilingSubgridsToWtiles(cu::Device& device, cu::Stream& stream,
                                const cu::Module& module,
                                const Parameters& parameters = {});

  void enqueue(int nr_subgrids, int nr_polarizations, long grid_size,
               int subgrid_size, int tile_size, int subgrid_offset,
               cu::DeviceMemory& d_metadata, cu::DeviceMemory& d_subgrid,
               cu::DeviceMemory& d_tiles, std::complex<float> scale);

  static constexpr unsigned kBlockSizeX = 128;
};

template <>
CompileDefinitions
KernelFactory<KernelWTilingSubgridsToWtiles>::compileDefinitions() const;

/*
  Wtiles from patch
*/
class KernelWTilingWTilesFromPatch : public CompiledKernel {
 public:
  static std::string source_file_;
  static std::string kernel_function_;

  KernelWTilingWTilesFromPatch(cu::Device& device, cu::Stream& stream,
                               const cu::Module& module,
                               const Parameters& parameters = {});

  void enqueue(int nr_polarizations, int nr_tiles, long grid_size,
               int tile_size, int padded_tile_size, int patch_size,
               idg::Coordinate patch_coordinate, cu::DeviceMemory& d_tile_ids,
               cu::DeviceMemory& d_tile_coordinates, cu::DeviceMemory& d_tiles,
               cu::DeviceMemory& d_patch);

  static constexpr unsigned kBlockSizeX = 128;
};

template <>
CompileDefinitions
KernelFactory<KernelWTilingWTilesFromPatch>::compileDefinitions() const;

/*
  Wtiles to patch
*/
class KernelWTilingWtilesToPatch : public CompiledKernel {
 public:
  static std::string source_file_;
  static std::string kernel_function_;

  KernelWTilingWtilesToPatch(cu::Device& device, cu::Stream& stream,
                             const cu::Module& module,
                             const Parameters& parameters = {});

  void enqueue(int nr_polarizations, int nr_tiles, long grid_size,
               int tile_size, int padded_tile_size, int patch_size,
               idg::Coordinate patch_coordinate, cu::DeviceMemory& d_tile_ids,
               cu::DeviceMemory& d_tile_coordinates, cu::DeviceMemory& d_tiles,
               cu::DeviceMemory& d_patch);

  static constexpr unsigned kBlockSizeX = 128;
};

template <>
CompileDefinitions
KernelFactory<KernelWTilingWtilesToPatch>::compileDefinitions() const;

/*
  Wtiles to grid
*/
class KernelWTilingWtilesToGrid : public CompiledKernel {
 public:
  static std::string source_file_;
  static std::string kernel_function_;

  KernelWTilingWtilesToGrid(cu::Device& device, cu::Stream& stream,
                            const cu::Module& module,
                            const Parameters& parameters = {});

  void enqueue(int nr_polarizations, int nr_tiles, long grid_size,
               int tile_size, int padded_tile_size,
               cu::DeviceMemory& d_tile_ids,
               cu::DeviceMemory& d_tile_coordinates, cu::DeviceMemory& d_tiles,
               cu::DeviceMemory& d_grid);

  static constexpr unsigned kBlockSizeX = 128;
};

template <>
CompileDefinitions
KernelFactory<KernelWTilingWtilesToGrid>::compileDefinitions() const;

/*
  Wtiles from grid
*/
class KernelWTilingWtilesFromGrid : public CompiledKernel {
 public:
  static std::string source_file_;
  static std::string kernel_function_;

  KernelWTilingWtilesFromGrid(cu::Device& device, cu::Stream& stream,
                              const cu::Module& module,
                              const Parameters& parameters = {});

  void enqueue(int nr_polarizations, int nr_tiles, long grid_size,
               int tile_size, int padded_tile_size,
               cu::DeviceMemory& d_tile_ids,
               cu::DeviceMemory& d_tile_coordinates, cu::DeviceMemory& d_tiles,
               cu::DeviceMemory& d_grid);

  static constexpr unsigned kBlockSizeX = 128;
};

template <>
CompileDefinitions
KernelFactory<KernelWTilingWtilesFromGrid>::compileDefinitions() const;

}  // namespace idg::kernel::cuda

#endif  // IDG_CUDA_KERNEL_WTILING_H_
