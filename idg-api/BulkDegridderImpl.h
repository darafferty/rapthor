// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * BulkDegridderImpl.h
 */

#ifndef IDG_BULKDEGRIDDERIMPL_H_
#define IDG_BULKDEGRIDDERIMPL_H_

#include "BulkDegridder.h"

#include "common/ArrayTypes.h"

namespace idg {
namespace api {

class BufferSetImpl;

class BulkDegridderImpl : public BulkDegridder {
 public:
  BulkDegridderImpl(const BufferSetImpl& bufferset,
                    const std::vector<double>& frequencies,
                    const std::size_t nr_stations);

  virtual ~BulkDegridderImpl();

  /** \brief Overridden from BulkDegridder */
  void compute_visibilities(
      const std::vector<size_t>& antennas1,
      const std::vector<size_t>& antennas2,
      const std::vector<const double*>& uvws,
      const std::vector<std::complex<float>*>& visibilities,
      const double* uvw_factors, const std::complex<float>* aterms,
      const std::vector<unsigned int>& aterm_offsets) const override;

 private:
  const BufferSetImpl& bufferset_;
  Array1D<float> frequencies_;
  std::size_t nr_stations_;
  Array1D<float> shift_;
};

}  // namespace api
}  // namespace idg

#endif
