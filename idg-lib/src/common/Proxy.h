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
  /*! Add visibilities to a grid, applying aterms
   *
   *  set_grid()
   *  Proxy::set_grid()
   *  idg::proxy::Proxy::set_grid()
   *  \verbatim embed:rst:leading-asterisk
   *  Before calling this function, the grid needs to have been set
   *  by a call to the set_grid function
   *  The plan can be obtained by a call to the make_plan function.
   *  :py:class:`idg.Proxy.Proxy`
   * :cpp:class:`Proxy <idg::proxy::Proxy>`
   *  \endverbatim
   */

  void gridding(const Plan& plan, const Array1D<float>& frequencies,
                const Array3D<Visibility<std::complex<float>>>& visibilities,
                const Array2D<UVW<float>>& uvw,
                const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
                const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                const Array1D<unsigned int>& aterms_offsets,
                const Array2D<float>& spheroidal);

  void degridding(
      const Plan& plan, const Array1D<float>& frequencies,
      Array3D<Visibility<std::complex<float>>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal);

  // Prepare a calibration cycle
  void calibrate_init(
      const unsigned int kernel_size, const Array1D<float>& frequencies,
      Array3D<Visibility<std::complex<float>>>& visibilities,
      Array3D<Visibility<float>>& weights, const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal);

  // Compute a hessian, gradient and residual, for the station with number
  // station_nr, given the current aterms and derivative aterms
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
  void transform(DomainAtoDomainB direction);

  void transform(DomainAtoDomainB direction, std::complex<float>* grid,
                 unsigned int grid_nr_correlations, unsigned int grid_height,
                 unsigned int grid_width);

  //! Computes the average beam term
  virtual void compute_avg_beam(
      const unsigned int nr_antennas, const unsigned int nr_channels,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array4D<float>& weights,
      idg::Array4D<std::complex<float>>& average_beam);

  //! Methods for querying and disabling Proxy capabilities
  bool supports_wstacking() {
    return (!m_disable_wstacking && do_supports_wstack_gridding() &&
            do_supports_wstack_degridding());
  }

  void set_disable_wstacking(bool v) { m_disable_wstacking = v; }

  virtual void set_disable_wtiling(bool v) { m_disable_wtiling = v; }

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

  /**
   * Set grid to be used for gridding, degridding or calibration
   */
  virtual void set_grid(std::shared_ptr<Grid> grid);

  /**
   * Flush all pending operations and return the current grid.
   */
  virtual std::shared_ptr<Grid> get_final_grid();

  /**
   * Get the current grid without flushing pending operations.
   * Use this function for reading the grid dimensions, and not the grid data.
   */
  const Grid& get_grid() const { return *m_grid; }

  //! Methods for cache management
  virtual void init_cache(int subgrid_size, float cell_size, float w_step,
                          const Array1D<float>& shift) {
    m_cache_state.subgrid_size = subgrid_size;
    m_cache_state.cell_size = cell_size;
    m_cache_state.w_step = w_step;
    m_cache_state.shift(0) = shift(0);
    m_cache_state.shift(1) = shift(1);
  };

  // Create a plan
  // The cache needs to have been initialized by call to init_cache first
  virtual std::unique_ptr<Plan> make_plan(
      const int kernel_size, const Array1D<float>& frequencies,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array1D<unsigned int>& aterms_offsets,
      Plan::Options options = Plan::Options()) {
    options.w_step = m_cache_state.w_step;
    return std::unique_ptr<Plan>(
        new Plan(kernel_size, m_cache_state.subgrid_size, m_grid->get_y_dim(),
                 m_cache_state.cell_size, m_cache_state.shift, frequencies, uvw,
                 baselines, aterms_offsets, options));
  }

 private:
  //! Degrid the visibilities from a uniform grid
  virtual void do_gridding(
      const Plan& plan, const Array1D<float>& frequencies,
      const Array3D<Visibility<std::complex<float>>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal) = 0;

  virtual void do_degridding(
      const Plan& plan, const Array1D<float>& frequencies,
      Array3D<Visibility<std::complex<float>>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal) = 0;

  // Uses rvalue references (&&) for all containers do_calibrate_init will take
  // ownership of. Call with std::move(...)
  virtual void do_calibrate_init(
      std::vector<std::unique_ptr<Plan>>&& plans,
      const Array1D<float>& frequencies,
      Array4D<Visibility<std::complex<float>>>&& visibilities,
      Array4D<Visibility<float>>&& weights, Array3D<UVW<float>>&& uvw,
      Array2D<std::pair<unsigned int, unsigned int>>&& baselines,
      const Array2D<float>& spheroidal) {}

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
  virtual void do_transform(DomainAtoDomainB direction){};

  virtual void do_compute_avg_beam(
      const unsigned int nr_antennas, const unsigned int nr_channels,
      const Array2D<UVW<float>>& uvw_array,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array4D<float>& weights,
      idg::Array4D<std::complex<float>>& average_beam);

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

  const int nr_polarizations = 4;

  std::vector<std::complex<float>> m_avg_aterm_correction;

 protected:
  virtual bool do_supports_wstack_gridding() { return false; }
  virtual bool do_supports_wstack_degridding() { return false; }
  virtual bool do_supports_wtiles() { return false; }

  std::shared_ptr<Grid> m_grid = nullptr;

  std::shared_ptr<Report> m_report;

  bool m_disable_wstacking = false;
  bool m_disable_wtiling = false;

  struct {
    int subgrid_size;
    float cell_size;
    float w_step;
    Array1D<float> shift{2};
  } m_cache_state;

};  // end class Proxy

}  // namespace proxy
}  // namespace idg

#endif
