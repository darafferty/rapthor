// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_CUDA_H_
#define IDG_CUDA_H_

#include <vector>
#include <complex>

#include "idg-common.h"

namespace idg {
namespace kernel {
namespace cuda {
class InstanceCUDA;
}
}  // namespace kernel

namespace proxy {
namespace cuda {
class CUDA : public Proxy {
 public:
  CUDA(ProxyInfo info);

  ~CUDA();

 public:
  std::unique_ptr<auxiliary::Memory> allocate_memory(size_t bytes) override;

  void print_compiler_flags();

  void print_devices();

  unsigned int get_num_devices() const;
  idg::kernel::cuda::InstanceCUDA& get_device(unsigned int i) const;

  std::vector<int> compute_jobsize(const Plan& plan,
                                   const unsigned int nr_stations,
                                   const unsigned int nr_timeslots,
                                   const unsigned int nr_timesteps,
                                   const unsigned int nr_channels,
                                   const unsigned int subgrid_size);

  void initialize(
      const Plan& plan, const float w_step, const Array1D<float>& shift,
      const float cell_size, const unsigned int kernel_size,
      const unsigned int subgrid_size, const Array1D<float>& frequencies,
      const Array3D<Visibility<std::complex<float>>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal);

  void cleanup();

 protected:
  void init_devices();
  void free_devices();
  static ProxyInfo default_info();

  struct {
    unsigned int nr_stations = 0;
    unsigned int nr_timeslots = 0;
    unsigned int nr_timesteps = 0;
    unsigned int nr_channels = 0;
    unsigned int subgrid_size = 0;
    unsigned int nr_baselines = 0;
    std::vector<int> jobsize;
    std::vector<int> max_nr_subgrids;
  } m_gridding_state;

  /*
   * Options used internally by the CUDA proxies
   */
  // Fraction of device memory reserved
  // for e.g. cuFFT. This memory is not taken
  // into account when computing  in compute_jobsize.
  float m_fraction_reserved = 0.15;

  // Use Unified Memory to store the grid, instead of having
  // a copy on the grid on the device.
  bool m_use_unified_memory = false;

  // Option to enable/disable reordering of the grid
  // to the host grid format, rather than the tiled
  // format used in the adder and splitter kernels.
  bool m_enable_tiling = true;

  // Maximum number of streams used to implement
  // multi-buffering to overlap I/O and computation
  unsigned int m_max_nr_streams = 2;

 public:
  void set_fraction_reserved(float f) { m_fraction_reserved = f; }
  void enable_unified_memory() { m_use_unified_memory = true; }

 protected:
  struct JobData {
    unsigned current_time_offset;
    unsigned current_nr_baselines;
    unsigned current_nr_subgrids;
    unsigned current_nr_timesteps;
    void* metadata_ptr;
    void* uvw_ptr;
    void* visibilities_ptr;
  };

  std::vector<JobData> jobs;

 private:
  ProxyInfo& mInfo;
  std::vector<idg::kernel::cuda::InstanceCUDA*> devices;
};
}  // namespace cuda
}  // end namespace proxy
}  // end namespace idg

#endif
