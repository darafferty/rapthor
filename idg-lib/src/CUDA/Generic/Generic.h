// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_CUDA_GENERIC_H_
#define IDG_CUDA_GENERIC_H_

#include "idg-common.h"
#include "CUDA/common/CUDA.h"

namespace powersensor {
class PowerSensor;
}

namespace idg {
namespace proxy {
namespace cuda {
/**
 * @brief Generic CUDA Proxy
 *
 */
class Generic : public CUDA {
  friend class Unified;

 public:
  // Constructor
  Generic(ProxyInfo info = default_info());

  // Destructor
  ~Generic();

 private:
  void do_gridding(
      const Plan& plan, const Array1D<float>& frequencies,
      const Array4D<std::complex<float>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal) override;

  void do_degridding(
      const Plan& plan, const Array1D<float>& frequencies,
      Array4D<std::complex<float>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal) override;

  void do_transform(DomainAtoDomainB direction) override;

  void run_imaging(
      const Plan& plan, const Array1D<float>& frequencies,
      const Array4D<std::complex<float>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      Grid& grid, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal, ImagingMode mode);

 public:
  bool do_supports_wtiling() override { return false; }

  void set_grid(std::shared_ptr<Grid> grid) override;

  std::shared_ptr<Grid> get_final_grid() override;

  virtual std::unique_ptr<Plan> make_plan(
      const int kernel_size, const Array1D<float>& frequencies,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array1D<unsigned int>& aterms_offsets,
      Plan::Options options) override;

  void init_cache(int subgrid_size, float cell_size, float w_step,
                  const Array1D<float>& shift) override;

 private:
  void check_grid();
  std::unique_ptr<cu::DeviceMemory> d_grid_;

  // W-Tiling
  void run_wtiles_to_grid(unsigned int subgrid_size, float image_size,
                          float w_step, const Array1D<float>& shift,
                          WTileUpdateInfo& wtile_flush_info);

  void run_subgrids_to_wtiles(unsigned int nr_polarizations,
                              unsigned int subgrid_offset,
                              unsigned int nr_subgrids,
                              unsigned int subgrid_size, float image_size,
                              float w_step, const Array1D<float>& shift,
                              WTileUpdateSet& wtile_flush_set,
                              cu::DeviceMemory& d_subgrids,
                              cu::DeviceMemory& d_metadata);

  void run_wtiles_from_grid(unsigned int subgrid_size, float image_size,
                            float w_step, const Array1D<float>& shift,
                            WTileUpdateInfo& wtile_initialize_info);

  void run_subgrids_from_wtiles(unsigned int nr_polarizations,
                                unsigned int subgrid_offset,
                                unsigned int nr_subgrids,
                                unsigned int subgrid_size, float image_size,
                                float w_step, const Array1D<float>& shift,
                                WTileUpdateSet& wtile_initialize_set,
                                cu::DeviceMemory& d_subgrids,
                                cu::DeviceMemory& d_metadata);

  void flush_wtiles();

  WTiles m_wtiles;
  unsigned int m_nr_tiles = 0;  // configured in init_cache
  const unsigned int m_tile_size = 128;
  const unsigned int m_patch_size = 512;
  const unsigned int m_nr_patches_batch = 3;
  struct {
    std::unique_ptr<cu::DeviceMemory> d_tiles;
    std::unique_ptr<cu::DeviceMemory> d_padded_tiles;
    std::unique_ptr<cu::HostMemory> h_tiles;
    std::vector<std::unique_ptr<cu::DeviceMemory>> d_patches;
  } m_buffers_wtiling;
};  // class Generic

}  // namespace cuda
}  // namespace proxy
}  // namespace idg
#endif