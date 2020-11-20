// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_PROXY2_H_
#define IDG_PROXY2_H_

#include <complex>
#include <vector>
#include <limits>
#include <cstring>
#include <utility>  // pair

#include "RuntimeWrapper.h"
#include "ProxyInfo.h"
#include "Types.h"
#include "Plan.h"
#include "Report.h"
#include "Exception.h"

namespace idg {
enum DomainAtoDomainB {
  FourierDomainToImageDomain,
  ImageDomainToFourierDomain
};

typedef std::string Compiler;
typedef std::string Compilerflags;
}  // namespace idg

namespace idg {
namespace proxy {

class Proxy {
 public:
  Proxy();
  virtual ~Proxy();

  /*
      High level routines
  */
  //! Grid the visibilities onto a uniform grid
  void gridding(const Plan& plan,
                const float w_step,  // in lambda
                const Array1D<float>& shift,
                const float cell_size,           // TODO: unit?
                const unsigned int kernel_size,  // full width in pixels
                const unsigned int subgrid_size,
                const Array1D<float>& frequencies,
                const Array3D<Visibility<std::complex<float>>>& visibilities,
                const Array2D<UVW<float>>& uvw,
                const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
                Grid& grid,
                const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                const Array1D<unsigned int>& aterms_offsets,
                const Array2D<float>& spheroidal);

  void gridding(const float w_step, const Array1D<float>& shift,
                const float cell_size, const unsigned int kernel_size,
                const unsigned int subgrid_size,
                const Array1D<float>& frequencies,
                const Array3D<Visibility<std::complex<float>>>& visibilities,
                const Array2D<UVW<float>>& uvw,
                const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
                Grid& grid,
                const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                const Array1D<unsigned int>& aterms_offsets,
                const Array2D<float>& spheroidal);

  void gridding(
      float w_step, float* shift, float cell_size, unsigned int kernel_size,
      unsigned int subgrid_size, float* frequencies, unsigned int nr_channels,
      std::complex<float>* visibilities, unsigned int visibilities_nr_baselines,
      unsigned int visibilities_nr_timesteps,
      unsigned int visibilities_nr_channels,
      unsigned int visibilities_nr_correlations, float* uvw,
      unsigned int uvw_nr_baselines, unsigned int uvw_nr_timesteps,
      unsigned int uvw_nr_coordinates,  // 3 (u, v, w)
      unsigned int* baselines, unsigned int baselines_nr_baselines,
      unsigned int baselines_two,  // antenna1, antenna2
      std::complex<float>* grid, unsigned int grid_nr_correlations,
      unsigned int grid_height, unsigned int grid_width,
      std::complex<float>* aterms, unsigned int aterms_nr_timeslots,
      unsigned int aterms_nr_stations, unsigned int aterms_aterm_height,
      unsigned int aterms_aterm_width, unsigned int aterms_nr_correlations,
      unsigned int* aterms_offsets,
      unsigned int aterms_offsets_nr_timeslots_plus_one, float* spheroidal,
      unsigned int spheroidal_height, unsigned int spheroidal_width);

  void degridding(
      const Plan& plan,
      const float w_step,  // in lambda
      const Array1D<float>& shift,
      const float cell_size,           // TODO: unit?
      const unsigned int kernel_size,  // full width in pixels
      const unsigned int subgrid_size, const Array1D<float>& frequencies,
      Array3D<Visibility<std::complex<float>>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Grid& grid, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal);

  void degridding(
      const float w_step, const Array1D<float>& shift, const float cell_size,
      const unsigned int kernel_size, const unsigned int subgrid_size,
      const Array1D<float>& frequencies,
      Array3D<Visibility<std::complex<float>>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Grid& grid, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal);

  void degridding(
      float w_step, float* shift, float cell_size, unsigned int kernel_size,
      unsigned int subgrid_size, float* frequencies, unsigned int nr_channels,
      std::complex<float>* visibilities, unsigned int visibilities_nr_baselines,
      unsigned int visibilities_nr_timesteps,
      unsigned int visibilities_nr_channels,
      unsigned int visibilities_nr_correlations, float* uvw,
      unsigned int uvw_nr_baselines, unsigned int uvw_nr_timesteps,
      unsigned int uvw_nr_coordinates,  // 3 (u, v, w)
      unsigned int* baselines, unsigned int baselines_nr_baselines,
      unsigned int baselines_two,  // antenna1, antenna2
      std::complex<float>* grid, unsigned int grid_nr_correlations,
      unsigned int grid_height, unsigned int grid_width,
      std::complex<float>* aterms, unsigned int aterms_nr_timeslots,
      unsigned int aterms_nr_stations, unsigned int aterms_aterm_height,
      unsigned int aterms_aterm_width, unsigned int aterms_nr_correlations,
      unsigned int* aterms_offsets,
      unsigned int aterms_offsets_nr_timeslots_plus_one, float* spheroidal,
      unsigned int spheroidal_height, unsigned int spheroidal_width);

  void calibrate_init(
      const float w_step, const Array1D<float>& shift, const float cell_size,
      const unsigned int kernel_size, const unsigned int subgrid_size,
      const Array1D<float>& frequencies,
      Array3D<Visibility<std::complex<float>>>& visibilities,
      Array3D<Visibility<float>>& weights, const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Grid& grid, const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal);

  void calibrate_update(
      const int station_nr,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array4D<Matrix2x2<std::complex<float>>>& derivative_aterms,
      Array3D<double>& hessian, Array2D<double>& gradient, double& residual);

  void calibrate_init_hessian_vector_product();

  void calibrate_update_hessian_vector_product1(
      const int station_nr,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array4D<Matrix2x2<std::complex<float>>>& derivative_aterms,
      const Array2D<float>& parameter_vector);

  void calibrate_update_hessian_vector_product2(
      const int station_nr,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array4D<Matrix2x2<std::complex<float>>>& derivative_aterms,
      Array2D<float>& parameter_vector);

  void calibrate_finish();

  //! Applies (inverse) Fourier transform to grid
  void transform(DomainAtoDomainB direction,
                 Array3D<std::complex<float>>& grid);

  void transform(DomainAtoDomainB direction, std::complex<float>* grid,
                 unsigned int grid_nr_correlations, unsigned int grid_height,
                 unsigned int grid_width);

  //! Methods for W-stacking
  bool supports_wstacking() {
    return (!m_disable_wstacking && do_supports_wstack_gridding() &&
            do_supports_wstack_degridding());
  }

  void set_disable_wstacking(bool v) { m_disable_wstacking = v; }

  void set_disable_wtiling(bool v) { m_disable_wtiling = v; }

  bool supports_wtiling() {
    return (!m_disable_wtiling && do_supports_wtiles());
  }

  virtual bool supports_avg_aterm_correction() { return false; }

  void set_avg_aterm_correction(
      const Array4D<std::complex<float>>& avg_aterm_correction);
  void unset_avg_aterm_correction();

  //! Methods for memory management
  virtual std::unique_ptr<auxiliary::Memory> allocate_memory(size_t bytes);

  template <typename T>
  Array1D<T> allocate_array1d(size_t a_dim) {
    auto bytes = a_dim * sizeof(T);
    return Array1D<T>(allocate_memory(bytes), a_dim);
  };

  template <typename T>
  Array2D<T> allocate_array2d(size_t b_dim, size_t a_dim) {
    auto bytes = a_dim * b_dim * sizeof(T);
    return Array2D<T>(allocate_memory(bytes), b_dim, a_dim);
  };

  template <typename T>
  Array3D<T> allocate_array3d(size_t c_dim, size_t b_dim, size_t a_dim) {
    auto bytes = a_dim * b_dim * c_dim * sizeof(T);
    return Array3D<T>(allocate_memory(bytes), c_dim, b_dim, a_dim);
  };

  template <typename T>
  Array4D<T> allocate_array4d(size_t d_dim, size_t c_dim, size_t b_dim,
                              size_t a_dim) {
    auto bytes = a_dim * b_dim * c_dim * d_dim * sizeof(T);
    return Array4D<T>(allocate_memory(bytes), d_dim, c_dim, b_dim, a_dim);
  };

  //! Methods for grid management
  virtual std::shared_ptr<Grid> allocate_grid(size_t nr_w_layers,
                                              size_t nr_correlations,
                                              size_t height, size_t width);

  virtual void set_grid(Grid& grid);
  virtual void set_grid(std::shared_ptr<Grid> grid);
  virtual std::shared_ptr<Grid> get_grid();

  //! Method W-tiling
  virtual std::unique_ptr<Plan> make_plan(
      const int kernel_size, const int subgrid_size, const int grid_size,
      const float cell_size, const Array1D<float>& frequencies,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array1D<unsigned int>& aterms_offsets,
      Plan::Options options = Plan::Options()) {
    return std::unique_ptr<Plan>(new Plan(kernel_size, subgrid_size, grid_size,
                                          cell_size, frequencies, uvw,
                                          baselines, aterms_offsets, options));
  }

 private:
  //! Degrid the visibilities from a uniform grid
  virtual void do_gridding(
      const Plan& plan,
      const float w_step,  // in lambda
      const Array1D<float>& shift,
      const float cell_size,           // TODO: unit?
      const unsigned int kernel_size,  // full width in pixels
      const unsigned int subgrid_size, const Array1D<float>& frequencies,
      const Array3D<Visibility<std::complex<float>>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      Grid& grid, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal) = 0;

  virtual void do_degridding(
      const Plan& plan,
      const float w_step,  // in lambda
      const Array1D<float>& shift,
      const float cell_size,           // TODO: unit?
      const unsigned int kernel_size,  // full width in pixels
      const unsigned int subgrid_size, const Array1D<float>& frequencies,
      Array3D<Visibility<std::complex<float>>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Grid& grid, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal) = 0;

  // Uses rvalue references (&&) for all containers do_calibrate_init will take
  // ownership of. Call with std::move(...)
  virtual void do_calibrate_init(
      std::vector<std::unique_ptr<Plan>>&& plans,
      float w_step,  // in lambda
      Array1D<float>&& shift,
      float cell_size,           // TODO: unit?
      unsigned int kernel_size,  // full width in pixels
      unsigned int subgrid_size, const Array1D<float>& frequencies,
      Array4D<Visibility<std::complex<float>>>&& visibilities,
      Array4D<Visibility<float>>&& weights, Array3D<UVW<float>>&& uvw,
      Array2D<std::pair<unsigned int, unsigned int>>&& baselines,
      const Grid& grid, const Array2D<float>& spheroidal) {}

  virtual void do_calibrate_update(
      const int station_nr,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array4D<Matrix2x2<std::complex<float>>>& derivative_aterms,
      Array3D<double>& hessian, Array2D<double>& gradient, double& residual) {}

  virtual void do_calibrate_finish() {}

  virtual void do_calibrate_init_hessian_vector_product() {}

  virtual void do_calibrate_update_hessian_vector_product1(
      const int station_nr,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array4D<Matrix2x2<std::complex<float>>>& derivative_aterms,
      const Array2D<float>& parameter_vector) {}

  virtual void do_calibrate_update_hessian_vector_product2(
      const int station_nr,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array4D<Matrix2x2<std::complex<float>>>& derivative_aterms,
      Array2D<float>& parameter_vector) {}

  //! Applyies (inverse) Fourier transform to grid
  virtual void do_transform(DomainAtoDomainB direction,
                            Array3D<std::complex<float>>& grid) = 0;

 protected:
  void check_dimensions(
      unsigned int subgrid_size, unsigned int frequencies_nr_channels,
      unsigned int visibilities_nr_baselines,
      unsigned int visibilities_nr_timesteps,
      unsigned int visibilities_nr_channels,
      unsigned int visibilities_nr_correlations, unsigned int uvw_nr_baselines,
      unsigned int uvw_nr_timesteps, unsigned int uvw_nr_coordinates,
      unsigned int baselines_nr_baselines, unsigned int baselines_two,
      unsigned int grid_nr_correlations, unsigned int grid_height,
      unsigned int grid_width, unsigned int aterms_nr_timeslots,
      unsigned int aterms_nr_stations, unsigned int aterms_aterm_height,
      unsigned int aterms_aterm_width, unsigned int aterms_nr_correlations,
      unsigned int aterms_offsets_nr_timeslots_plus_one,
      unsigned int spheroidal_height, unsigned int spheroidal_width) const;

  void check_dimensions(
      unsigned int subgrid_size, const Array1D<float>& frequencies,
      const Array3D<Visibility<std::complex<float>>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Grid& grid, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal) const;

  Array1D<float> compute_wavenumbers(const Array1D<float>& frequencies) const;

  const unsigned int nr_polarizations = 4;

  std::vector<std::complex<float>> m_avg_aterm_correction;

 protected:
  virtual bool do_supports_wstack_gridding() { return false; }
  virtual bool do_supports_wstack_degridding() { return false; }
  virtual bool do_supports_wtiles() { return false; }

  std::shared_ptr<Grid> m_grid = nullptr;

  Report report;

  bool m_disable_wstacking = false;
  bool m_disable_wtiling = false;

};  // end class Proxy

}  // namespace proxy
}  // namespace idg

#endif
