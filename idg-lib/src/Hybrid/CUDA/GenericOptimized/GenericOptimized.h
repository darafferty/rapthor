// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_HYBRID_GENERIC_OPTIMIZED_H_
#define IDG_HYBRID_GENERIC_OPTIMIZED_H_

#include <thread>

#include "idg-cpu.h"
#include "CUDA/common/CUDA.h"

namespace idg {
namespace proxy {
namespace hybrid {

class GenericOptimized : public cuda::CUDA {
 public:
  GenericOptimized();
  ~GenericOptimized();

  virtual bool do_supports_wstack_gridding() {
    return cpuProxy->do_supports_wstack_gridding();
  }
  virtual bool do_supports_wstack_degridding() {
    return cpuProxy->do_supports_wstack_degridding();
  }

  virtual bool do_supports_wtiles() override {
    return cpuProxy->do_supports_wtiles();
  }

  virtual bool supports_avg_aterm_correction() { return true; }

  void set_grid(std::shared_ptr<Grid> grid) override;

  std::shared_ptr<Grid> get_final_grid() override;

  void init_cache(int subgrid_size, float cell_size, float w_step,
                  const Array1D<float>& shift) override;

  std::unique_ptr<Plan> make_plan(
      const int kernel_size, const Array1D<float>& frequencies,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array1D<unsigned int>& aterms_offsets,
      Plan::Options options) override;

 private:
  void do_gridding(
      const Plan& plan, const Array1D<float>& frequencies,
      const Array3D<Visibility<std::complex<float>>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal) override;

  void do_degridding(
      const Plan& plan, const Array1D<float>& frequencies,
      Array3D<Visibility<std::complex<float>>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal) override;

  void do_transform(DomainAtoDomainB direction) override;

  void run_gridding(
      const Plan& plan, const Array1D<float>& frequencies,
      const Array3D<Visibility<std::complex<float>>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      Grid& grid, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal);

  void run_degridding(
      const Plan& plan, const Array1D<float>& frequencies,
      Array3D<Visibility<std::complex<float>>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Grid& grid, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal);

  void do_calibrate_init(
      std::vector<std::unique_ptr<Plan>>&& plans,
      const Array1D<float>& frequencies,
      Array4D<Visibility<std::complex<float>>>&& visibilities,
      Array4D<Visibility<float>>&& weights, Array3D<UVW<float>>&& uvw,
      Array2D<std::pair<unsigned int, unsigned int>>&& baselines,
      const Array2D<float>& spheroidal) override;

  void do_calibrate_update(
      const int station_nr,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array4D<Matrix2x2<std::complex<float>>>& derivative_aterms,
      Array3D<double>& hessian, Array2D<double>& gradient,
      double& residual) override;

  void do_calibrate_finish() override;

  void do_calibrate_init_hessian_vector_product() override;

  void do_calibrate_update_hessian_vector_product1(
      const int station_nr,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array4D<Matrix2x2<std::complex<float>>>& derivative_aterms,
      const Array2D<float>& parameter_vector) override;

  void do_calibrate_update_hessian_vector_product2(
      const int station_nr,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array4D<Matrix2x2<std::complex<float>>>& derivative_aterms,
      Array2D<float>& parameter_vector) override;

 protected:
  idg::proxy::cpu::CPU* cpuProxy;

  /*
   * Calibration state
   */
  struct {
    std::vector<std::unique_ptr<Plan>> plans;
    unsigned int nr_baselines;
    unsigned int nr_timesteps;
    unsigned int nr_channels;
    Array3D<UVW<float>> uvw;
    std::vector<unsigned int> d_sums_ids;
    unsigned int d_lmnp_id;
    std::vector<unsigned int> d_metadata_ids;
    std::vector<unsigned int> d_subgrids_ids;
    std::vector<unsigned int> d_visibilities_ids;
    std::vector<unsigned int> d_weights_ids;
    std::vector<unsigned int> d_uvw_ids;
    std::vector<unsigned int> d_aterm_idx_ids;
    Array3D<Visibility<std::complex<float>>>
        hessian_vector_product_visibilities;
  } m_calibrate_state;

  // Note:
  //     kernel_calibrate internally assumes max_nr_terms = 8
  //     and will process larger values of nr_terms in batches
  const unsigned int m_calibrate_max_nr_terms = 8;

};  // class GenericOptimized

}  // namespace hybrid
}  // namespace proxy
}  // namespace idg

#endif
