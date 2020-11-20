// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_CUDA_INSTANCE_H_
#define IDG_CUDA_INSTANCE_H_

#include <memory>

#include "idg-common.h"

#include "CU.h"
#include "CUFFT.h"
#include "PowerRecord.h"

namespace idg {
namespace kernel {
namespace cuda {

class InstanceCUDA : public KernelsInstance {
 public:
  // Constructor
  InstanceCUDA(ProxyInfo& info, int device_nr = 0, int device_id = 0);

  // Destructor
  ~InstanceCUDA();

  cu::Context& get_context() const { return *context; }
  cu::Device& get_device() const { return *device; }
  cu::Stream& get_execute_stream() const { return *executestream; };
  cu::Stream& get_htod_stream() const { return *htodstream; };
  cu::Stream& get_dtoh_stream() const { return *dtohstream; };

  void set_context() const { context->setCurrent(); }

  std::string get_compiler_flags();

  powersensor::State measure();
  void measure(PowerRecord& record, cu::Stream& stream);

  void launch_gridder(int time_offset, int nr_subgrids, int grid_size,
                      int subgrid_size, float image_size, float w_step,
                      int nr_channels, int nr_stations, cu::DeviceMemory& d_uvw,
                      cu::DeviceMemory& d_wavenumbers,
                      cu::DeviceMemory& d_visibilities,
                      cu::DeviceMemory& d_spheroidal, cu::DeviceMemory& d_aterm,
                      cu::DeviceMemory& d_aterm_indices,
                      cu::DeviceMemory& d_avg_aterm_correction,
                      cu::DeviceMemory& d_metadata,
                      cu::DeviceMemory& d_subgrid);

  void launch_degridder(
      int time_offset, int nr_subgrids, int grid_size, int subgrid_size,
      float image_size, float w_step, int nr_channels, int nr_stations,
      cu::DeviceMemory& d_uvw, cu::DeviceMemory& d_wavenumbers,
      cu::DeviceMemory& d_visibilities, cu::DeviceMemory& d_spheroidal,
      cu::DeviceMemory& d_aterm, cu::DeviceMemory& d_aterm_indices,
      cu::DeviceMemory& d_metadata, cu::DeviceMemory& d_subgrid);

  void launch_calibrate(
      int nr_subgrids, int grid_size, int subgrid_size, float image_size,
      float w_step, int total_nr_timesteps, int nr_channels, int nr_stations,
      int nr_terms, cu::DeviceMemory& d_uvw, cu::DeviceMemory& d_wavenumbers,
      cu::DeviceMemory& d_visibilities, cu::DeviceMemory& d_weights,
      cu::DeviceMemory& d_aterm, cu::DeviceMemory& d_aterm_derivatives,
      cu::DeviceMemory& d_aterm_indices, cu::DeviceMemory& d_metadata,
      cu::DeviceMemory& d_subgrid, cu::DeviceMemory& d_sums1,
      cu::DeviceMemory& d_sums2, cu::DeviceMemory& d_lmnp,
      cu::DeviceMemory& d_hessian, cu::DeviceMemory& d_gradient,
      cu::DeviceMemory& d_residual);

  void launch_grid_fft(cu::DeviceMemory& d_data, int size,
                       DomainAtoDomainB direction);

  void plan_subgrid_fft(unsigned size, unsigned batch);

  void launch_subgrid_fft(cu::DeviceMemory& d_data, unsigned nr_subgrids,
                          DomainAtoDomainB direction);

  void launch_grid_fft_unified(unsigned long size, unsigned batch,
                               Array3D<std::complex<float>>& grid,
                               DomainAtoDomainB direction);

  void launch_adder(int nr_subgrids, long grid_size, int subgrid_size,
                    cu::DeviceMemory& d_metadata, cu::DeviceMemory& d_subgrid,
                    cu::DeviceMemory& d_grid);

  void launch_adder_unified(int nr_subgrids, long grid_size, int subgrid_size,
                            cu::DeviceMemory& d_metadata,
                            cu::DeviceMemory& d_subgrid, void* u_grid);

  void launch_splitter(int nr_subgrids, long grid_size, int subgrid_size,
                       cu::DeviceMemory& d_metadata,
                       cu::DeviceMemory& d_subgrid, cu::DeviceMemory& d_grid);

  void launch_splitter_unified(int nr_subgrids, long grid_size,
                               int subgrid_size, cu::DeviceMemory& d_metadata,
                               cu::DeviceMemory& d_subgrid, void* u_grid);

  void launch_scaler(int nr_subgrids, int subgrid_size,
                     cu::DeviceMemory& d_subgrid);

  void launch_scaler(int nr_subgrids, int subgrid_size, void* u_subgrid);

  // Memory management per device
  cu::DeviceMemory& allocate_device_grid(size_t bytes);
  cu::DeviceMemory& allocate_device_wavenumbers(size_t bytes);
  cu::DeviceMemory& allocate_device_aterms(size_t bytes);
  cu::DeviceMemory& allocate_device_aterms_indices(size_t bytes);
  cu::DeviceMemory& allocate_device_spheroidal(size_t bytes);
  cu::DeviceMemory& allocate_device_avg_aterm_correction(size_t bytes);

  // Memory management per stream
  cu::HostMemory& allocate_host_subgrids(size_t bytes);
  cu::HostMemory& allocate_host_visibilities(size_t bytes);
  cu::HostMemory& allocate_host_uvw(size_t bytes);
  cu::DeviceMemory& allocate_device_visibilities(unsigned int id, size_t bytes);
  cu::DeviceMemory& allocate_device_uvw(unsigned int id, size_t bytes);
  cu::DeviceMemory& allocate_device_subgrids(unsigned int id, size_t bytes);
  cu::DeviceMemory& allocate_device_metadata(unsigned int id, size_t bytes);

  // Memory management for misc device buffers
  unsigned int allocate_device_memory(size_t bytes);
  cu::DeviceMemory& retrieve_device_memory(unsigned int id);

  // Memory management for misc page-locked host buffers
  void register_host_memory(void* ptr, size_t bytes);

  // Retrieve pre-allocated buffers (per device)
  cu::DeviceMemory& retrieve_device_grid() { return *d_grid.get(); }
  cu::DeviceMemory& retrieve_device_aterms() { return *d_aterms; }
  cu::DeviceMemory& retrieve_device_aterms_indices() {
    return *d_aterms_indices;
  }
  cu::DeviceMemory& retrieve_device_aterms_derivatives() {
    return *d_aterms_derivatives;
  }
  cu::DeviceMemory& retrieve_device_wavenumbers() { return *d_wavenumbers; }
  cu::DeviceMemory& retrieve_device_spheroidal() { return *d_spheroidal; }
  cu::DeviceMemory& retrieve_device_avg_aterm_correction() {
    return *d_avg_aterm_correction;
  }

  // Retrieve pre-allocated buffers (per stream)
  cu::DeviceMemory& retrieve_device_visibilities(unsigned int id) {
    return *d_visibilities_[id];
  }
  cu::DeviceMemory& retrieve_device_uvw(unsigned int id) { return *d_uvw_[id]; }
  cu::DeviceMemory& retrieve_device_subgrids(unsigned int id) {
    return *d_subgrids_[id];
  }
  cu::DeviceMemory& retrieve_device_metadata(unsigned int id) {
    return *d_metadata_[id];
  }

  // Free buffers
  void free_device_wavenumbers() { d_wavenumbers.reset(); };
  void free_device_spheroidal() { d_spheroidal.reset(); };
  void free_device_aterms() { d_aterms.reset(); };
  void free_device_aterms_indices() { d_aterms_indices.reset(); };
  void free_device_avg_aterm_correction() { d_avg_aterm_correction.reset(); };
  void free_device_visibilities() { d_visibilities_.clear(); };
  void free_device_uvw() { d_uvw_.clear(); };
  void free_device_subgrids() { d_subgrids_.clear(); };
  void free_device_metadata() { d_metadata_.clear(); };
  void unmap_host_memory() { h_registered_.clear(); };

  // Misc
  void free_fft_plans();
  int get_tile_size_grid() const { return tile_size_grid; };
  void print_device_memory_info();

 private:
  void free_host_memory();
  void free_device_memory();
  void reset();

 protected:
  cu::Module* compile_kernel(std::string& flags, std::string& src,
                             std::string& bin);
  void compile_kernels();
  void load_kernels();
  void set_parameters();
  void set_parameters_default();
  void set_parameters_kepler();
  void set_parameters_maxwell();
  void set_parameters_gp100();
  void set_parameters_pascal();
  void set_parameters_volta();

 protected:
  // Variables shared by all InstanceCUDA instances
  ProxyInfo& mInfo;

 private:
  std::unique_ptr<cu::Context> context;
  std::unique_ptr<cu::Device> device;
  std::unique_ptr<cu::Stream> executestream;
  std::unique_ptr<cu::Stream> htodstream;
  std::unique_ptr<cu::Stream> dtohstream;
  std::unique_ptr<cu::Function> function_gridder;
  std::unique_ptr<cu::Function> function_degridder;
  std::unique_ptr<cu::Function> function_fft;
  std::unique_ptr<cu::Function> function_adder;
  std::unique_ptr<cu::Function> function_splitter;
  std::unique_ptr<cu::Function> function_scaler;
  std::vector<std::unique_ptr<cu::Function>> functions_calibrate;

  // One instance per device
  std::unique_ptr<cu::DeviceMemory> d_aterms;
  std::unique_ptr<cu::DeviceMemory> d_aterms_indices;
  std::unique_ptr<cu::DeviceMemory> d_aterms_derivatives;
  std::unique_ptr<cu::DeviceMemory> d_avg_aterm_correction;
  std::unique_ptr<cu::DeviceMemory> d_wavenumbers;
  std::unique_ptr<cu::DeviceMemory> d_spheroidal;
  std::unique_ptr<cu::DeviceMemory> d_grid;
  std::unique_ptr<cu::HostMemory> h_visibilities;
  std::unique_ptr<cu::HostMemory> h_uvw;
  std::unique_ptr<cu::HostMemory> h_subgrids;

  // One instance per stream
  std::vector<std::unique_ptr<cu::DeviceMemory>> d_visibilities_;
  std::vector<std::unique_ptr<cu::DeviceMemory>> d_uvw_;
  std::vector<std::unique_ptr<cu::DeviceMemory>> d_metadata_;
  std::vector<std::unique_ptr<cu::DeviceMemory>> d_subgrids_;

  // Registered host memory
  std::vector<std::unique_ptr<cu::RegisteredMemory>> h_registered_;

  // Misc device memory
  std::vector<std::unique_ptr<cu::DeviceMemory>> d_misc_;

  // All CUDA modules private to this InstanceCUDA
  std::vector<std::unique_ptr<cu::Module>> mModules;

 protected:
  dim3 block_gridder;
  dim3 block_degridder;
  dim3 block_calibrate;
  dim3 block_adder;
  dim3 block_splitter;
  dim3 block_scaler;

  int batch_gridder;
  int batch_degridder;
  int tile_size_grid;

  // Grid FFT
  int m_fft_grid_size = 0;
  std::unique_ptr<cufft::C2C_2D> m_fft_plan_grid;

  // Subgrid FFT
  const unsigned m_fft_subgrid_bulk_default = 1024;
  unsigned m_fft_subgrid_bulk = m_fft_subgrid_bulk_default;
  unsigned m_fft_subgrid_size = 0;
  std::unique_ptr<cufft::C2C_2D> m_fft_plan_subgrid;
  std::unique_ptr<cu::DeviceMemory> d_fft_subgrid;

 private:
  // Memory allocation/reuse methods
  template <typename T>
  T* reuse_memory(uint64_t size, std::unique_ptr<T>& memory);

  template <typename T>
  T* reuse_memory(std::vector<std::unique_ptr<T>>& memories, unsigned int id,
                  uint64_t size);

  template <typename T>
  T* reuse_memory(std::vector<std::unique_ptr<T>>& memories, uint64_t size,
                  void* ptr);

 public:
  void enqueue_report(cu::Stream& stream, int nr_timesteps, int nr_subgrids);

  void copy_htoh(void* dst, void* src, size_t bytes);

  void copy_dtoh(cu::Stream& stream, void* dst, cu::DeviceMemory& src,
                 size_t bytes);

  void copy_htod(cu::Stream& stream, cu::DeviceMemory& dst, void* src,
                 size_t bytes);

 private:
  void start_measurement(void* data);
  void end_measurement(void* data);
};
std::ostream& operator<<(std::ostream& os, InstanceCUDA& d);

// Kernel names
static const std::string name_gridder = "kernel_gridder";
static const std::string name_degridder = "kernel_degridder";
static const std::string name_adder = "kernel_adder";
static const std::string name_splitter = "kernel_splitter";
static const std::string name_fft = "kernel_fft";
static const std::string name_scaler = "kernel_scaler";
static const std::string name_calibrate_lmnp = "kernel_calibrate_lmnp";
static const std::string name_calibrate_sums = "kernel_calibrate_sums";
static const std::string name_calibrate_gradient = "kernel_calibrate_gradient";
static const std::string name_calibrate_hessian = "kernel_calibrate_hessian";

}  // end namespace cuda
}  // end namespace kernel
}  // end namespace idg

#endif
