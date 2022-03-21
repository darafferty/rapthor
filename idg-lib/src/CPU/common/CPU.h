// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_CPU_H_
#define IDG_CPU_H_

#include "idg-common.h"

#include "InstanceCPU.h"

namespace idg {
namespace proxy {
namespace cpu {

class CPU : public Proxy {
 public:
  // Constructor
  CPU();

  // Disallow assignment and pass-by-value
  CPU& operator=(const CPU& rhs) = delete;
  CPU(const CPU& v) = delete;

  // Destructor
  virtual ~CPU();

  std::unique_ptr<auxiliary::Memory> allocate_memory(size_t bytes) override;

  virtual bool do_supports_wstacking() override {
    return m_kernels->do_supports_wstacking();
  }

  virtual bool do_supports_wtiling() override {
    return m_kernels->do_supports_wtiling();
  }

  std::shared_ptr<kernel::cpu::InstanceCPU> get_kernels() { return m_kernels; }

  std::unique_ptr<Plan> make_plan(
      const int kernel_size, const Array1D<float>& frequencies,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array1D<unsigned int>& aterms_offsets,
      Plan::Options options) override;

  void init_cache(int subgrid_size, float cell_size, float w_step,
                  const Array1D<float>& shift) override;

  std::shared_ptr<Grid> get_final_grid() override;

 private:
  unsigned int compute_jobsize(const Plan& plan,
                               const unsigned int nr_timesteps,
                               const unsigned int nr_channels,
                               const unsigned int nr_correlations,
                               const unsigned int nr_polarizations,
                               const unsigned int subgrid_size);

  // Routines
  void do_gridding(
      const Plan& plan, const Array1D<float>& frequencies,
      const Array4D<std::complex<float>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal) override;

  void do_degridding(
      const Plan& plan, const Array1D<float>& frequencies,
      Array4D<std::complex<float>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal) override;

  void do_calibrate_init(
      std::vector<std::vector<std::unique_ptr<Plan>>>&& plans,
      const Array2D<float>& frequencies,
      Array6D<std::complex<float>>&& visibilities, Array6D<float>&& weights,
      Array3D<UVW<float>>&& uvw,
      Array2D<std::pair<unsigned int, unsigned int>>&& baselines,
      const Array2D<float>& taper) override;

  virtual void do_calibrate_update(
      const int station_nr,
      const Array5D<Matrix2x2<std::complex<float>>>& aterms,
      const Array5D<Matrix2x2<std::complex<float>>>& derivative_aterms,
      Array4D<double>& hessian, Array3D<double>& gradient,
      Array1D<double>& residual) override;

  void do_calibrate_finish() override;

  void do_transform(DomainAtoDomainB direction) override;

  void do_compute_avg_beam(
      const unsigned int nr_antennas, const unsigned int nr_channels,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array4D<float>& weights,
      idg::Array4D<std::complex<float>>& average_beam) override;

 protected:
  void init_wtiles(int grid_size, int subgrid_size, float image_size,
                   float w_step);

  std::shared_ptr<kernel::cpu::InstanceCPU> m_kernels;
  std::shared_ptr<powersensor::PowerSensor> m_powersensor;

  /*
   * Options used internally by the CPU proxy
   */
  // Maximum fraction of available memory used to allocate subgrids
  // this value impacts the jobsize that will be used and hence the
  // amount of memory additionaly allocated (if any) in various kernels.
  float m_fraction_memory_subgrids = 0.10;

  // Maximum size of the subgrids buffer allocated in do_gridding
  // and do_degridding. A value of about 10x the size of the L3 cache
  // seems to provide a good balance between the number of kernel calls
  // and the time needed to allocate memory, while it is large enough
  // to provide sufficient scalability.
  size_t m_max_bytes_subgrids = 512 * 1024 * 1024;  // 512 Mb

  WTiles m_wtiles;

  struct {
    std::vector<std::vector<std::unique_ptr<Plan>>> plans;
    unsigned int nr_baselines;
    unsigned int nr_timesteps;
    unsigned int nr_channels;
    Array1D<float> wavenumbers;
    Array6D<std::complex<float>> visibilities;  // ANTxANTxTIMExCHANxCOR
    Array6D<float> weights;                     // ANTxANTxTIMExCHANxCOR
    Array3D<UVW<float>> uvw;
    Array2D<std::pair<unsigned int, unsigned int>> baselines;
    std::vector<Array4D<std::complex<float>>> subgrids;
    std::vector<Array4D<std::complex<float>>> phasors;
    std::vector<int> max_nr_timesteps;
  } m_calibrate_state;

};  // end class CPU

}  // end namespace cpu
}  // end namespace proxy
}  // end namespace idg
#endif
