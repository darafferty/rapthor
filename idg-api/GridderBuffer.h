// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * GridderBuffer.h
 *
 * \class GridderBuffer
 *
 * \brief Access to IDG's high level gridder routines
 *
 * The GridderBuffer manages a buffer of a fixed number of time steps
 * One fills the buffer, which fill occasionally be flushed to grid
 * the visibilities onto the grid.
 *
 * Usage (pseudocode):
 *
 * idg::GridderBuffer plan(...);
 * plan.set_grid(grid);
 * plan.set_other_properties(...);
 * plan.bake();
 *
 * for (auto row = 0; row < nr_rows; ++row) {
 *    gridder.grid_visibilities(...);
 * }
 *
 * // Make sure no visibilites are still in the buffer
 * gridder.finished();
 *
 * // Transform the gridded visibilities to an image
 * gridder.transform_grid();
 *
 */

#ifndef IDG_GRIDDERBUFFER_H_
#define IDG_GRIDDERBUFFER_H_

#include <complex>

#include "Buffer.h"

namespace idg {
namespace api {

class GridderBuffer : public virtual Buffer {
 public:
  // Constructors and destructor
  virtual ~GridderBuffer(){};

  /** \brief Adds the visibilities to the buffer
   *  \param timeIndex [in] 0 <= timeIndex < NR_TIMESTEPS
   *                        or 0 <= timeIndex < bufferTimesteps
   *  \param antenna1 [in]  0 <= antenna1 < nrStations
   *  \param antenna2 [in]  antenna1 < antenna2 < nrStations
   *  \param uvwInMeters [in] double[3]: (u, v, w)
   *  \param visibilities [in]
   * std::complex<float>[NR_CHANNELS][NR_POLARIZATIONS]
   */
  virtual void grid_visibilities(size_t timeIndex, size_t antenna1,
                                 size_t antenna2, const double* uvwInMeters,
                                 std::complex<float>* visibilities,
                                 const float* weights) = 0;

 protected:
  GridderBuffer() {}
};

}  // namespace api
}  // namespace idg

#endif
