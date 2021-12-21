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

#ifndef IDG_GRIDDERBUFFERIMPL_H_
#define IDG_GRIDDERBUFFERIMPL_H_

#include <complex>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <thread>

#include "idg-common.h"
#if defined(BUILD_LIB_CPU)
#include "idg-cpu.h"
#endif

#include "GridderBuffer.h"
#include "BufferImpl.h"

namespace idg {
namespace api {

class BufferSetImpl;

class GridderBufferImpl : public virtual GridderBuffer, public BufferImpl {
 public:
  // Constructors and destructor
  GridderBufferImpl(const BufferSetImpl &bufferset, size_t bufferTimesteps);

  virtual ~GridderBufferImpl();

  /** \brief Adds the visibilities to the buffer
   *  \param timeIndex [in] 0 <= timeIndex < NR_TIMESTEPS
   *                        or 0 <= timeIndex < bufferTimesteps
   *  \param antenna1 [in]  0 <= antenna1 < nrStations
   *  \param antenna2 [in]  antenna1 < antenna2 < nrStations
   *  \param uvwInMeters [in] double[3]: (u, v, w)
   *  \param visibilities [in]
   * std::complex<float>[NR_CHANNELS][NR_CORRELATIONS]
   */
  void grid_visibilities(size_t timeIndex, size_t antenna1, size_t antenna2,
                         const double *uvwInMeters,
                         const std::complex<float> *visibilities,
                         const float *weights);

  /** \brief Configure computing average beams.
   *  \param beam Pointer to average beam data. If the pointer is null,
   *         average beam computations are disabled.
   */
  void set_avg_beam(std::complex<float> *average_beam) {
    m_average_beam = average_beam;
  }

  /** \brief Computes average beam
   */
  void compute_avg_beam();

  /** \brief Signal that not more visibilies are gridded */
  virtual void finished() override;

  /** \brief Explicitly flush the buffer */
  virtual void flush() override;

  /** reset_aterm() Resets the new aterm for the next time chunk */
  virtual void reset_aterm();

 protected:
  virtual void malloc_buffers();

 private:
  // secondary buffers
  Array2D<UVW<float>> m_bufferUVW2;  // BL x TI
  Array1D<std::pair<unsigned int, unsigned int>> m_bufferStationPairs2;  // BL
  Array4D<std::complex<float>> m_bufferVisibilities2;     // BL x TI x CH x CR
  std::vector<Matrix2x2<std::complex<float>>> m_aterms2;  // ST x SB x SB
  Array4D<float> m_buffer_weights;   // BL x TI x NR_CHANNELS x NR_CORRELATIONS
  Array4D<float> m_buffer_weights2;  // BL x TI x NR_CHANNELS x NR_CORRELATIONS
  std::vector<unsigned int> m_aterm_offsets2;

  std::thread m_flush_thread;
  void flush_thread_worker();

  // Pointer to average beam data in the parent BufferSet.
  // If it is null, compute_avg_beam() will not run.
  std::complex<float> *m_average_beam;
};

}  // namespace api
}  // namespace idg

#endif
