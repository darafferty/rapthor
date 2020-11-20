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
  CPU(std::vector<std::string> libraries);

  // Disallow assignment and pass-by-value
  CPU& operator=(const CPU& rhs) = delete;
  CPU(const CPU& v) = delete;

  // Destructor
  virtual ~CPU();

  std::unique_ptr<auxiliary::Memory> allocate_memory(size_t bytes) override;

  virtual bool do_supports_wstack_gridding() override {
    return kernels.has_adder_wstack();
  }
  virtual bool do_supports_wstack_degridding() override {
    return kernels.has_splitter_wstack();
  }
  virtual bool supports_avg_aterm_correction() override { return true; }

  virtual bool do_supports_wtiles() override {
    return kernels.has_adder_wtiles() && kernels.has_splitter_wtiles();
  }

  kernel::cpu::InstanceCPU& get_kernels() { return kernels; }

  virtual std::unique_ptr<Plan> make_plan(
      const int kernel_size, const int subgrid_size, const int grid_size,
      const float cell_size, const Array1D<float>& frequencies,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array1D<unsigned int>& aterms_offsets,
      Plan::Options options) override;

  virtual void set_grid(Grid& grid) override;
  virtual void set_grid(std::shared_ptr<Grid> grid) override;
  virtual std::shared_ptr<Grid> get_grid() override;

 private:
  unsigned int compute_jobsize(const Plan& plan,
                               const unsigned int nr_timesteps,
                               const unsigned int nr_channels,
                               const unsigned int subgrid_size);

  // Routines
  virtual void do_gridding(
      const Plan& plan,
      const float w_step,  // in lambda
      const Array1D<float>& shift, const float cell_size,
      const unsigned int kernel_size,  // full width in pixels
      const unsigned int subgrid_size, const Array1D<float>& frequencies,
      const Array3D<Visibility<std::complex<float>>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      Grid& grid, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal) override;

  virtual void do_degridding(
      const Plan& plan,
      const float w_step,  // in lambda
      const Array1D<float>& shift, const float cell_size,
      const unsigned int kernel_size,  // full width in pixels
      const unsigned int subgrid_size, const Array1D<float>& frequencies,
      Array3D<Visibility<std::complex<float>>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Grid& grid, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal) override;

  virtual void do_calibrate_init(
      std::vector<std::unique_ptr<Plan>>&& plans,
      float w_step,  // in lambda
      Array1D<float>&& shift, float cell_size,
      unsigned int kernel_size,  // full width in pixels
      unsigned int subgrid_size, const Array1D<float>& frequencies,
      Array4D<Visibility<std::complex<float>>>&& visibilities,
      Array4D<Visibility<float>>&& weights, Array3D<UVW<float>>&& uvw,
      Array2D<std::pair<unsigned int, unsigned int>>&& baselines,
      const Grid& grid, const Array2D<float>& spheroidal) override;

  virtual void do_calibrate_update(
      const int station_nr,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array4D<Matrix2x2<std::complex<float>>>& derivative_aterms,
      Array3D<double>& hessian, Array2D<double>& gradient,
      double& residual) override;

  virtual void do_calibrate_finish() override;

  virtual void do_calibrate_init_hessian_vector_product() override;

  virtual void do_calibrate_update_hessian_vector_product1(
      const int station_nr,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array4D<Matrix2x2<std::complex<float>>>& derivative_aterms,
      const Array2D<float>& parameter_vector) override;

  virtual void do_calibrate_update_hessian_vector_product2(
      const int station_nr,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array4D<Matrix2x2<std::complex<float>>>& derivative_aterms,
      Array2D<float>& parameter_vector) override;

  virtual void do_transform(DomainAtoDomainB direction,
                            Array3D<std::complex<float>>& grid) override;

 protected:
  void init_wtiles(int grid_size, int subgrid_size, float image_size,
                   float w_step);

  kernel::cpu::InstanceCPU kernels;
  powersensor::PowerSensor* powerSensor;

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
    std::vector<std::unique_ptr<Plan>> plans;
    float w_step;  // in lambda
    Array1D<float> shift;
    float cell_size;
    float image_size;
    unsigned int kernel_size;
    unsigned int grid_size;
    unsigned int subgrid_size;
    unsigned int nr_baselines;
    unsigned int nr_timesteps;
    unsigned int nr_channels;
    Array1D<float> wavenumbers;
    Array4D<Visibility<std::complex<float>>> visibilities;
    Array4D<Visibility<float>> weights;
    Array3D<UVW<float>> uvw;
    Array2D<std::pair<unsigned int, unsigned int>> baselines;
    std::vector<Array4D<std::complex<float>>> subgrids;
    std::vector<Array4D<std::complex<float>>> phasors;
    std::vector<int> max_nr_timesteps;
    Array3D<Visibility<std::complex<float>>>
        hessian_vector_product_visibilities;
  } m_calibrate_state;

};  // end class CPU

}  // end namespace cpu
}  // end namespace proxy
}  // end namespace idg
#endif
