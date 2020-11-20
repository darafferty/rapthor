// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

/*
 * BufferImpl.h
 * Access to IDG's high level gridder routines
 */

#ifndef IDG_BUFFERIMPL_H_
#define IDG_BUFFERIMPL_H_

#include <complex>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include "idg-common.h"
#if defined(BUILD_LIB_CPU)
#include "idg-cpu.h"
#endif
#if defined(BUILD_LIB_CUDA)
#include "idg-cuda.h"
#endif
#if defined(BUILD_LIB_CUDA) && defined(BUILD_LIB_CPU)
#include "idg-hybrid-cuda.h"
#endif
#if defined(BUILD_LIB_OPENCL)
#include "idg-opencl.h"
#endif
#include "Datatypes.h"

#include "Buffer.h"

namespace idg {
namespace api {

class BufferSetImpl;

class BufferImpl : public virtual Buffer {
 public:
  // Constructors and destructor
  BufferImpl(const BufferSetImpl& bufferset, size_t bufferTimesteps = 4096);

  virtual ~BufferImpl();

  // Set/get all parameters
  void set_frequencies(size_t channelCount, const double* frequencyList);

  void set_frequencies(const std::vector<double>& frequency_list);

  size_t get_frequencies_size() const;
  double get_frequency(size_t channel) const;

  void set_stations(size_t nrStations);
  size_t get_stations() const;

  size_t get_nr_polarizations() const;

  void set_image(double* image) {}

  // Bake the plan after parameters are set
  // Must be called before the plan is used
  // if have settings have changed after construction
  void bake();

  // Flush the buffer explicitly
  virtual void flush() = 0;

  virtual void finished() {}

  /** \brief Sets a new aterm for the buffer
   *  \param timeIndex [in] 0 <= timeIndex < NR_TIMESTEPS
   *                        or 0 <= timeIndex < bufferTimesteps
   *  \param aterm [in]
   * std::complex<float>[nrStations][subgridsize][subgridsize]
   */
  virtual void set_aterm(size_t timeIndex, const std::complex<float>* aterms);

  double get_image_size() const;

 protected:
  /* Helper function to map (antenna1, antenna2) -> baseline index
   * The baseline index is formed such that:
   *   0 implies antenna1=0, antenna2=1 ;
   *   1 implies antenna1=0, antenna2=2 ;
   * n-1 implies antenna1=1, antenna2=2 etc. */
  size_t baseline_index(size_t antenna1, size_t antenna2) const;

  // Other helper routines
  virtual void malloc_buffers();
  void reset_buffers();
  void set_uvw_to_infinity();
  void init_default_aterm();

  const BufferSetImpl& m_bufferset;  // Reference to parent BufferSet.

  // Bookkeeping
  size_t m_bufferTimesteps;
  size_t m_timeStartThisBatch;
  size_t m_timeStartNextBatch;
  std::set<size_t> m_timeindices;

  // Parameters for proxy
  size_t m_nrStations;
  size_t m_nr_channels;
  size_t m_nr_baselines;
  size_t m_nrPolarizations;
  Array1D<float> m_shift;
  std::vector<unsigned int> m_default_aterm_offsets;
  std::vector<unsigned int> m_aterm_offsets;
  Array1D<unsigned int> m_aterm_offsets_array;

  // Buffers
  Array1D<float> m_frequencies;  // CH
  std::vector<Matrix2x2<std::complex<float>>> m_aterms;
  std::vector<Matrix2x2<std::complex<float>>> m_default_aterms;
  Array4D<Matrix2x2<std::complex<float>>> m_aterms_array;  // ST x SB x SB

  Array2D<UVW<float>> m_bufferUVW;  // BL x TI
  Array1D<std::pair<unsigned int, unsigned int>> m_bufferStationPairs;  // BL
  Array3D<Visibility<std::complex<float>>>
      m_bufferVisibilities;  // BL x TI x CH
};

}  // namespace api
}  // namespace idg

#endif
