// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * BulkDegridder.h
 *
 * \class BulkDegridder
 *
 * \brief Degridder that computes visibilities for a range of input data in one
 * go.
 */

#ifndef IDG_BULKDEGRIDDER_H_
#define IDG_BULKDEGRIDDER_H_

#include <complex>
#include <vector>

namespace idg {
namespace api {

class BulkDegridder {
 public:
  // Destructor
  virtual ~BulkDegridder(){};

  /**
   * Compute visibilities for multiple baselines and timesteps.
   * This function combines the functionality of the request_visibilities()
   * and compute() functions of the DegridderBuffer class.
   * @param antennas1 [in] First antenna for each baseline.
   * @param antennas2 [in] Second antenna for each baseline. antennas2[bl] must
   *        be larger than antennas1[bl] for all baselines bl.
   * @param uvws [in] Vector with pointers to uvw values.
   *        The vector length should equal the number of time steps.
   *        The pointers hold values for all baselines in that time step.
   * @param visibilities [out] Vector with pointers to the visibilities.
   *        The vector length should equal the number of time steps.
   *        The pointers hold values for all baselines in that time step.
   * @param uvw_factors [in] Multiplication factors for uvw values.
   *        This function multiplies all uvw input values by these factors.
   *        If the pointer is null, (1.0, 1.0, 1.0) is used.
   * @param aterms [in] Pointer to one or more blocks with aterms. The block
   *        size is nr_stations * subgrid_size^2 * nr_correlations.
   *        If the pointer is null, default aterms are used.
   * @param aterm_offsets [in] Time offsets for applying the aterms.
   *        For example { 0, 4 } means:
   *        - Use the first aterm block for time steps 0, 1, 2, and 3.
   *        - Use the second aterm block for time steps 4 and above.
   *        'aterms' should thus contain at least two blocks in this case.
   *        The first time offset must always be zero.
   *        The vector may not be empty.
   */
  virtual void compute_visibilities(
      const std::vector<size_t>& antennas1,
      const std::vector<size_t>& antennas2,
      const std::vector<const double*>& uvws,
      const std::vector<std::complex<float>*>& visibilities,
      const double* uvw_factors = nullptr,
      const std::complex<float>* aterms = nullptr,
      const std::vector<unsigned int>& aterm_offsets = {0}) const = 0;
};

}  // namespace api
}  // namespace idg

#endif
