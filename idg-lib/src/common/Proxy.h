// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_PROXY2_H_
#define IDG_PROXY2_H_

#include <complex>
#include <vector>
#include <limits>
#include <cstring>
#include <utility>  // pair

#include <aocommon/xt/span.h>

#include "RuntimeWrapper.h"
#include "ProxyInfo.h"
#include "Types.h"
#include "Plan.h"
#include "Report.h"
#include "Exception.h"
#include "Tensor.h"

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

  /**
   * @brief Add visibilities to a grid, applying A-terms.
   *
   * Before calling this function, the grid needs to have been set
   * by a call to the set_grid function and the cache needs to be initialized
   * by a call to the init_cache function.
   * On return the results might still be in a cache, and to
   * obtain the final grid a call to get_final_grid() is needed.
   *
   * @param[in] plan A Plan object, previously created by a call to the
   * make_plan() member function. The Plan contains the partitioning of the
   * visibilities in subgrids.
   * @param[in] frequencies A one dimensional array of floats, containing the
   * frequency per channel
   * @param[in] visibilities A four dimensional array of complex floats. The
   * axes are baseline, time, channel, and correlation (XX,XY,YX,XX). Note that
   * the baseline and time axis are transposed compared to the ordering in a
   * Measurement Set.
   * @param[in] uvw A two dimensional array of UVW coordinates (triplet of
   * floats). The axes are baseline and time.
   * @param[in] baselines A one dimensional array of pairs station indices
   * (integers) comprising a baseline.
   * @param[in] aterms Four dimensional array of 2x2 (Jones) matrices of complex
   * floats. The axes are time slot, station, subgrid x, subgrid y. A time slot
   * is a range of time samples over which the aterms are constant The time
   * slots are defined by the aterm_offsets parameters.
   * @param[in] aterm_offsets A one dimensional array of time indices (ints)
   * that represent the time ranges of time slots. The array is one longer than
   * the number of time slots. Time slot k is valid from aterm_offsets[k] until
   * aterm_offsets[k+1].
   * @param[in] taper A two dimensional array of floats of the size of a
   * subgrid.
   */
  void gridding(
      const Plan& plan, const aocommon::xt::Span<float, 1>& frequencies,
      const aocommon::xt::Span<std::complex<float>, 4>& visibilities,
      const aocommon::xt::Span<UVW<float>, 2>& uvw,
      const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
          baselines,
      const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
      const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
      const aocommon::xt::Span<float, 2>& taper);

  /**
   * @brief Degrid (predict) visibilities, applying A-terms.
   *
   * Before calling this function, the grid needs to have been set
   * by a call to the set_grid function, and the cache needs to be initialized
   * by a call to the init_cache function.
   *
   * @param[in] plan A Plan object, previously created by a call to the
   * make_plan() member function. The Plan contains the partitioning of the
   * visibilities in subgrids.
   * @param[in] frequencies A one dimensional array of floats, containing the
   * frequency per channel
   * @param[out] visibilities A four dimensional array of complex floats. The
   * axes are baseline, time, channel, and correlation (XX,XY,YX,XX). Note that
   * the baseline and time axis are transposed compared to the ordering in a
   * Measurement Set.
   * @param[in] uvw A two dimensional array of UVW coordinates (triplet of
   * floats). The axes are baseline and time.
   * @param[in] baselines A one dimensional array of pairs station indices
   * (integers) comprising a baseline.
   * @param[in] aterms Four dimensional array of 2x2 (Jones) matrices of complex
   * floats. The axes are time slot, station, subgrid x, subgrid y. A time slot
   * is a range of time samples over which the aterms are constant The time
   * slots are defined by the aterm_offsets parameters.
   * @param[in] aterm_offsets A one dimensional array of time indices (ints)
   * that represent the time ranges of time slots. The array is one longer than
   * the number of time slots. Time slot k is valid from aterm_offsets[k] until
   * aterm_offsets[k+1].
   * @param[in] taper A two dimensional array of floats of the size of a
   * subgrid.
   */
  void degridding(
      const Plan& plan, const aocommon::xt::Span<float, 1>& frequencies,
      aocommon::xt::Span<std::complex<float>, 4>& visibilities,
      const aocommon::xt::Span<UVW<float>, 2>& uvw,
      const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
          baselines,
      const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
      const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
      const aocommon::xt::Span<float, 2>& taper);

  /**
   * @brief Prepare a calibration cycle
   *
   * @param kernel_size Size of the kernel, see
   * \verbatim embed:rst:leading-asterisk
   * :doc:`kernelsize`
   * \endverbatim
   * @param[in] frequencies A nr_channel_blocks x nr_channels_per_block array
   * of floats, containing the frequency per channel.
   * @param[in] visibilities A four dimensional array of complex floats. The
   * axes are baseline, time, channel, and correlation (XX,XY,YX,XX). Note that
   * the baseline and time axis are transposed compared to the ordering in a
   * Measurement Set.
   * @param weights Visibility weights, should have same dimension as \c
   * visibilities.
   * @param[in] uvw A two dimensional array of UVW coordinates (triplet of
   * floats). The axes are baseline and time.
   * @param[in] baselines A one dimensional array of pairs station indices
   * (integers) comprising a baseline.
   * @param[in] aterm_offsets A one dimensional array of time indices (ints)
   * that represent the time ranges of time slots. The array is one longer than
   * the number of time slots. Time slot k is valid from aterm_offsets[k] until
   * aterm_offsets[k+1].
   * @param[in] taper A two dimensional array of floats of the size of a
   * subgrid.
   */
  void calibrate_init(
      const unsigned int kernel_size,
      const aocommon::xt::Span<float, 2>& frequencies,
      aocommon::xt::Span<std::complex<float>, 4>& visibilities,
      aocommon::xt::Span<float, 4>& weights,
      const aocommon::xt::Span<UVW<float>, 2>& uvw,
      const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
          baselines,
      const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
      const aocommon::xt::Span<float, 2>& taper);

  /**
   * @brief Compute a hessian, gradient and residual per time slot for station
   * antenna_nr, given the current aterms and derivative aterms.
   *
   * The calibration functions provided by the Proxy do not implement a full
   * solving strategy. They are intended to be called by an iterative solver to
   * compute the Hessian, derivative and residual in some working point.
   *
   * It is assumed that the solver updates the aterm of a single station at a
   * time, and iterates over the stations. The aterm is assumed to be described
   * by a model with a number of unknown parameters.
   *
   * The aterm_derivatives are the derivatives of the aterm of station
   * antenna_nr with respect to the unknowns.
   *
   * The values returned are
   * 1) the residual, the root mean square (TODO check, maybe it is the sum
   * of squares) of
   * the difference between the predicted (degridded) visibilities and the
   * visibilities provided in the calibrate_init() call. 2) the gradient, the
   * derivative of the residual with respect to the unknowns 3) the derivative
   * of the gradient with respect to the unknowns
   *
   * @param[in] antenna_nr
   * @param[in] aterms Five dimensional array of 2x2 (Jones) matrices of complex
   * floats. The axes are channel block, time slot, station, subgrid x, subgrid
   * y. A time slot is a range of time samples over which the aterms are
   * constant The time slots are defined by the aterm_offsets parameters.
   * @param[in] aterm_derivatives Five dimensional array of 2x2 (Jones) matrices
   * of complex floats. The axes are channel block, time slot, terms, subgrid x,
   * subgrid y.
   *
   *
   * @param[out] hessian
   * @param[out] gradient
   * @param[out] residual
   */
  void calibrate_update(
      const int antenna_nr,
      const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 5>& aterms,
      const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 5>&
          aterm_derivatives,
      aocommon::xt::Span<double, 4>& hessian,
      aocommon::xt::Span<double, 3>& gradient,
      aocommon::xt::Span<double, 1>& residual);

  /**
   * @brief Clean up after calibration cycle.
   */
  void calibrate_finish();

  //! Applies (inverse) Fourier transform to grid
  void transform(DomainAtoDomainB direction);

  void transform(DomainAtoDomainB direction, std::complex<float>* grid,
                 unsigned int grid_nr_correlations, unsigned int grid_height,
                 unsigned int grid_width);

  /**
   * @brief Computes the average beam term
   *
   * @param[in] nr_antennas
   * @param[in] nr_channels
   * @param[in] uvw
   * @param[in] baselines
   * @param[in] aterms
   * @param[in] aterm_offsets
   * @param[in] weights
   * @param[out] average_beam Four dimensional array of complex floats.
   *                          The axes are subgrid x, subgrid y, mueller matrix
   * row, mueller matrix col
   */
  virtual void compute_avg_beam(
      const unsigned int nr_antennas, const unsigned int nr_channels,
      const aocommon::xt::Span<UVW<float>, 2>& uvw,
      const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
          baselines,
      const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
      const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
      const aocommon::xt::Span<float, 4>& weights,
      aocommon::xt::Span<std::complex<float>, 4>& average_beam);

  //! Methods for querying and disabling Proxy capabilities
  bool supports_wstacking() {
    return (!m_disable_wstacking && do_supports_wstacking());
  }

  void set_disable_wstacking(bool v) { m_disable_wstacking = v; }

  virtual void set_disable_wtiling(bool v) { m_disable_wtiling = v; }

  bool supports_wtiling() {
    return (!m_disable_wtiling && do_supports_wtiling());
  }

  void set_avg_aterm_correction(
      const aocommon::xt::Span<std::complex<float>, 4>& avg_aterm_correction);
  void unset_avg_aterm_correction();

  //! Methods for memory management
  virtual std::unique_ptr<auxiliary::Memory> allocate_memory(size_t bytes);

  template <typename T, size_t Dimensions>
  Tensor<T, Dimensions> allocate_tensor(
      const std::initializer_list<size_t> shape) {
    assert(shape.size() == Dimensions);
    std::array<size_t, Dimensions> shape_array;
    size_t bytes = sizeof(T);
    for (size_t i = 0; i < shape.size(); i++) {
      const size_t dimension = *(shape.begin() + i);
      shape_array[i] = dimension;
      bytes *= dimension;
    }
    return Tensor<T, Dimensions>(allocate_memory(bytes), shape_array);
  }

  template <typename T, size_t Dimensions>
  aocommon::xt::Span<T, Dimensions> allocate_span(
      const std::initializer_list<size_t> shape) {
    assert(shape.size() == Dimensions);
    std::array<size_t, Dimensions> shape_array;
    size_t bytes = sizeof(T);
    for (size_t i = 0; i < shape.size(); i++) {
      const size_t dimension = *(shape.begin() + i);
      shape_array[i] = dimension;
      bytes *= dimension;
    }
    std::unique_ptr<auxiliary::Memory> memory = allocate_memory(bytes);
    T* ptr = reinterpret_cast<T*>(memory->data());
    memory_.push_back(std::move(memory));
    return aocommon::xt::CreateSpan(ptr, shape_array);
  }

  /**
   * Set grid to be used for gridding, degridding or calibration.
   */
  virtual void set_grid(aocommon::xt::Span<std::complex<float>, 4>& grid);
  virtual void free_grid();

  /**
   * @brief Flush all pending operations and return the final grid.
   * @return aocommon::xt::Span<std::complex<float>
   */
  virtual aocommon::xt::Span<std::complex<float>, 4>& get_final_grid();

  /**
   * @brief Get the current grid without flushing pending operations.
   *
   * @return aocommon::xt::Span<std::complex<float>
   */
  aocommon::xt::Span<std::complex<float>, 4>& get_grid() { return grid_; }

  //! Methods for cache management

  /**
   * @brief Initialize cache
   *
   * Sets the configuration for subsequent gridding, degridding or calibrate
   * calls This allows the Proxy to set up a caching strategy
   *
   * @param subgrid_size Size of the subgrid, see
   * \verbatim embed:rst:leading-asterisk
   * :doc:`kernelsize`
   * \endverbatim

   * @param cell_size
   * @param w_step
   * @param shift
   */
  virtual void init_cache(int subgrid_size, float cell_size, float w_step,
                          const std::array<float, 2>& shift) {
    m_cache_state.subgrid_size = subgrid_size;
    m_cache_state.cell_size = cell_size;
    m_cache_state.w_step = w_step;
    m_cache_state.shift = shift;
  };

  // The cache needs to have been initialized by call to init_cache first
  /**
   * @brief Create a plan that can be used for calls to the gridding or
   * degridding functions
   *
   * @param kernel_size Size of the kernel, see
   * \verbatim embed:rst:leading-asterisk
   * :doc:`kernelsize`
   * \endverbatim
   * @param[in] frequencies A one dimensional array of floats, containing the
   * frequency per channel
   * @param[in] uvw A two dimensional array of UVW coordinates (triplet of
   * floats). The axes are baseline and time.
   * @param[in] baselines A one dimensional array of pairs station indices
   * (integers) comprising a baseline.
   * @param[in] aterm_offsets A one dimensional array of time indices (ints)
   * that represent the time ranges of time slots. The array is one longer than
   * the number of time slots. Time slot k is valid from aterm_offsets[k] until
   * aterm_offsets[k+1].
   * @param options
   * @return std::unique_ptr<Plan>
   */
  virtual std::unique_ptr<Plan> make_plan(
      const int kernel_size, const aocommon::xt::Span<float, 1>& frequencies,
      const aocommon::xt::Span<UVW<float>, 2>& uvw,
      const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
          baselines,
      const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
      Plan::Options options = Plan::Options()) {
    options.w_step = m_cache_state.w_step;
    const size_t grid_size = get_grid().shape(2);
    assert(get_grid().shape(3) == grid_size);
    return std::make_unique<Plan>(kernel_size, m_cache_state.subgrid_size,
                                  grid_size, m_cache_state.cell_size,
                                  m_cache_state.shift, frequencies, uvw,
                                  baselines, aterm_offsets, options);
  }

 private:
  //! Degrid the visibilities from a uniform grid
  virtual void do_gridding(
      const Plan& plan, const aocommon::xt::Span<float, 1>& frequencies,
      const aocommon::xt::Span<std::complex<float>, 4>& visibilities,
      const aocommon::xt::Span<UVW<float>, 2>& uvw,
      const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
          baselines,
      const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
      const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
      const aocommon::xt::Span<float, 2>& taper) = 0;

  virtual void do_degridding(
      const Plan& plan, const aocommon::xt::Span<float, 1>& frequencies,
      aocommon::xt::Span<std::complex<float>, 4>& visibilities,
      const aocommon::xt::Span<UVW<float>, 2>& uvw,
      const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
          baselines,
      const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
      const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
      const aocommon::xt::Span<float, 2>& taper) = 0;

  // Using rvalue references (&&) for all containers do_calibrate_init will take
  // ownership of. Call with std::move(...)
  virtual void do_calibrate_init(
      std::vector<std::vector<std::unique_ptr<Plan>>>&& plans,
      const aocommon::xt::Span<float, 2>& frequencies,
      Tensor<std::complex<float>, 6>&& visibilities, Tensor<float, 6>&& weights,
      Tensor<UVW<float>, 3>&& uvw,
      Tensor<std::pair<unsigned int, unsigned int>, 2>&& baselines,
      const aocommon::xt::Span<float, 2>& taper) {
    throw std::runtime_error(
        "do_calibrate_init is not implemented by this proxy");
  }

  virtual void do_calibrate_update(
      const int antenna_nr,
      const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 5>& aterms,
      const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 5>&
          aterm_derivatives,
      aocommon::xt::Span<double, 4>& hessian,
      aocommon::xt::Span<double, 3>& gradient,
      aocommon::xt::Span<double, 1>& residual) {
    throw std::runtime_error(
        "do_calibrate_update is not implemented by this proxy");
  }

  virtual void do_calibrate_finish() {}

  //! Applyies (inverse) Fourier transform to grid
  virtual void do_transform(DomainAtoDomainB direction){};

  virtual void do_compute_avg_beam(
      const unsigned int nr_antennas, const unsigned int nr_channels,
      const aocommon::xt::Span<UVW<float>, 2>& uvw,
      const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
          baselines,
      const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
      const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
      const aocommon::xt::Span<float, 4>& weights,
      aocommon::xt::Span<std::complex<float>, 4>& average_beam);

 protected:
  void check_dimensions(
      const Plan::Options& options, unsigned int subgrid_size,
      unsigned int frequencies_nr_channels,
      unsigned int visibilities_nr_baselines,
      unsigned int visibilities_nr_timesteps,
      unsigned int visibilities_nr_channels,
      unsigned int visibilities_nr_correlations, unsigned int uvw_nr_baselines,
      unsigned int uvw_nr_timesteps, unsigned int uvw_nr_coordinates,
      unsigned int baselines_nr_baselines, unsigned int baselines_two,
      unsigned int grid_nr_polarizations, unsigned int grid_height,
      unsigned int grid_width, unsigned int aterms_nr_timeslots,
      unsigned int aterms_nr_stations, unsigned int aterms_aterm_height,
      unsigned int aterms_aterm_width, unsigned int aterms_nr_polarizations,
      unsigned int aterm_offsets_nr_timeslots_plus_one,
      unsigned int taper_height, unsigned int taper_width) const;

  void check_dimensions(
      const Plan::Options& options, unsigned int subgrid_size,
      const aocommon::xt::Span<float, 1>& frequencies,
      const aocommon::xt::Span<std::complex<float>, 4>& visibilities,
      const aocommon::xt::Span<UVW<float>, 2>& uvw,
      const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
          baselines,
      const aocommon::xt::Span<std::complex<float>, 4>& grid,
      const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
      const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
      const aocommon::xt::Span<float, 2>& taper) const;

  Tensor<float, 1> compute_wavenumbers(
      const aocommon::xt::Span<float, 1>& frequencies);

  const int nr_correlations = 4;

  aocommon::xt::Span<std::complex<float>, 4> m_avg_aterm_correction;

 protected:
  virtual bool do_supports_wstacking() { return false; }
  virtual bool do_supports_wtiling() { return false; }

  bool m_disable_wstacking = false;
  bool m_disable_wtiling = false;

  struct {
    int subgrid_size;
    float cell_size;
    float w_step;
    std::array<float, 2> shift;
  } m_cache_state;

  void free_memory() { memory_.clear(); };

  std::shared_ptr<Report>& get_report() { return report_; }

 private:
  std::shared_ptr<Report> report_;
  std::vector<std::unique_ptr<auxiliary::Memory>> memory_;
  aocommon::xt::Span<std::complex<float>, 4> grid_;

};  // end class Proxy

}  // namespace proxy
}  // namespace idg

#endif
