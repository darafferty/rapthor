// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_KERNELS_CPU_H_
#define IDG_KERNELS_CPU_H_

#include <cstdint>
#include <ccomplex>
#include <cmath>
#include <string>
#include <memory>  // unique_ptr

#include "idg-common.h"

namespace idg {
namespace kernel {
namespace cpu {

class InstanceCPU : public KernelsInstance {
 public:
  static constexpr int kWTileSize = 128;

  // Constructor
  InstanceCPU(std::vector<std::string> libraries);

  // Destructor
  virtual ~InstanceCPU();

  void run_gridder(int nr_subgrids, int grid_size, int subgrid_size,
                   float image_size, float w_step, const float *shift,
                   int nr_channels, int nr_stations, void *uvw,
                   void *wavenumbers, void *visibilities, void *spheroidal,
                   void *aterm, void *aterm_idx, void *avg_aterm,
                   void *metadata, void *subgrid);

  void run_degridder(int nr_subgrids, int grid_size, int subgrid_size,
                     float image_size, float w_step, const float *shift,
                     int nr_channels, int nr_stations, void *uvw,
                     void *wavenumbers, void *visibilities, void *spheroidal,
                     void *aterm, void *aterm_idx, void *metadata,
                     void *subgrid);

  void run_average_beam(int nr_baselines, int nr_antennas, int nr_timesteps,
                        int nr_channels, int nr_aterms, int subgrid_size,
                        void *uvw, void *baselines, void *aterms,
                        void *aterms_offsets, void *weights,
                        void *average_beam);

  void run_calibrate(int nr_subgrids, int grid_size, int subgrid_size,
                     float image_size, float w_step, const float *shift,
                     int max_nr_timesteps, int nr_channels, int nr_terms,
                     int nr_stations, int nr_time_slots, const UVW<float> *uvw,
                     const float *wavenumbers, float2 *visibilities,
                     const float *weights, const float2 *aterm,
                     const float2 *aterm_derivative, const int *aterms_indices,
                     const Metadata *metadata, const float2 *subgrid,
                     const float2 *phasors, double *hessian, double *gradient,
                     double *residual);

  void run_calibrate_hessian_vector_product1(
      const int station_nr,
      const Array4D<Matrix2x2<std::complex<float>>> &aterms,
      const Array4D<Matrix2x2<std::complex<float>>> &derivative_aterms,
      const Array2D<float> &parameter_vector);

  void run_calibrate_hessian_vector_product2(
      const int station_nr,
      const Array4D<Matrix2x2<std::complex<float>>> &aterms,
      const Array4D<Matrix2x2<std::complex<float>>> &derivative_aterms,
      Array2D<float> &parameter_vector);

  void run_phasor(int nr_subgrids, int grid_size, int subgrid_size,
                  float image_size, float w_step, const float *shift,
                  int max_nr_timesteps, int nr_channels, void *uvw,
                  void *wavenumbers, void *metadata, void *phasors);

  void run_fft(int grid_size, int size, int batch, void *data, int direction);

  void run_subgrid_fft(int grid_size, int size, int batch, void *data,
                       int direction);

  void run_adder(int nr_subgrids, int grid_size, int subgrid_size,
                 void *metadata, void *subgrid, void *grid);

  void run_splitter(int nr_subgrids, int grid_size, int subgrid_size,
                    void *metadata, void *subgrid, void *grid);

  void run_adder_wstack(int nr_subgrids, int grid_size, int subgrid_size,
                        void *metadata, void *subgrid, void *grid);

  void run_splitter_wstack(int nr_subgrids, int grid_size, int subgrid_size,
                           void *metadata, void *subgrid, void *grid);

  void run_adder_subgrids_to_wtiles(int nr_subgrids, int grid_size,
                                    int subgrid_size, void *metadata,
                                    void *subgrid);

  void run_splitter_subgrids_from_wtiles(int nr_subgrids, int grid_size,
                                         int subgrid_size, void *metadata,
                                         void *subgrid);

  void run_adder_wtiles_to_grid(int grid_size, int subgrid_size,
                                float image_size, float w_step,
                                const float *shift, int nr_tiles, int *tile_ids,
                                idg::Coordinate *tile_coordinates,
                                std::complex<float> *grid);

  void run_adder_wtiles_to_grid(int nr_tiles, int wtile_size,
                                int w_padded_tile_size, int grid_size,
                                idg::Coordinate *tile_coordinates,
                                std::complex<float> *tiles,
                                std::complex<float> *grid);

  void run_splitter_wtiles_from_grid(int grid_size, int subgrid_size,
                                     float image_size, float w_step,
                                     const float *shift, int nr_tiles,
                                     int *tile_ids,
                                     Coordinate *tile_coordinates,
                                     std::complex<float> *grid);

  void run_splitter_wtiles_to_grid(int nr_tiles, int wtile_size,
                                   int w_padded_tile_size, int grid_size,
                                   idg::Coordinate *tile_coordinates,
                                   std::complex<float> *tiles,
                                   std::complex<float> *grid);

  void run_adder_wtiles(unsigned int nr_subgrids, unsigned int grid_size,
                        unsigned int subgrid_size, float image_size,
                        float w_step, const float *shift, int subgrid_offset,
                        WTileUpdateSet &wtile_flush_set, void *metadata,
                        void *subgrid, std::complex<float> *grid);

  void run_splitter_wtiles(int nr_subgrids, int grid_size, int subgrid_size,
                           float image_size, float w_step, const float *shift,
                           int subgrid_offset,
                           WTileUpdateSet &wtile_initialize_set, void *metadata,
                           void *subgrid, std::complex<float> *grid);

  bool has_adder_wstack() { return (function_adder_wstack != nullptr); }

  bool has_splitter_wstack() { return (function_splitter_wstack != nullptr); }

  bool has_adder_wtiles() {
    return (function_adder_subgrids_to_wtiles != nullptr) &&
           (function_adder_wtiles_to_grid != nullptr);
  }

  bool has_splitter_wtiles() {
    return (function_splitter_wtiles_from_grid != nullptr) &&
           (function_splitter_subgrids_from_wtiles != nullptr);
  }

  /**
   * Creates the buffer to store the wtiles
   *
   * The size of buffer in number of wtiles is determined by a heuristic based
   * on grid_size and wtile size kWTileSize.
   *
   * @param grid_size size of the grid
   * @param subgrid_size size of the subgrids
   * @return The number of wtiles
   */
  virtual size_t init_wtiles(size_t grid_size, int subgrid_size);

 protected:
  void compile(Compiler compiler, Compilerflags flags);
  void load_shared_objects(std::vector<std::string> libraries);
  void load_kernel_funcions();

  std::vector<runtime::Module *> modules;

  idg::Array1D<std::complex<float>> m_wtiles_buffer;

  std::unique_ptr<runtime::Function> function_gridder;
  std::unique_ptr<runtime::Function> function_degridder;
  std::unique_ptr<runtime::Function> function_calibrate;
  std::unique_ptr<runtime::Function> function_calibrate_hessian_vector_product1;
  std::unique_ptr<runtime::Function> function_calibrate_hessian_vector_product2;
  std::unique_ptr<runtime::Function> function_phasor;
  std::unique_ptr<runtime::Function> function_fft;
  std::unique_ptr<runtime::Function> function_adder;
  std::unique_ptr<runtime::Function> function_splitter;
  std::unique_ptr<runtime::Function> function_adder_wstack;
  std::unique_ptr<runtime::Function> function_splitter_wstack;
  std::unique_ptr<runtime::Function> function_adder_wtiles_to_grid;
  std::unique_ptr<runtime::Function> function_splitter_wtiles_from_grid;
  std::unique_ptr<runtime::Function> function_adder_subgrids_to_wtiles;
  std::unique_ptr<runtime::Function> function_splitter_subgrids_from_wtiles;
  std::unique_ptr<runtime::Function> function_tiles_to_grid;
  std::unique_ptr<runtime::Function> function_tiles_from_grid;
  std::unique_ptr<runtime::Function> function_average_beam;
};

// Kernel names
static const std::string name_gridder = "kernel_gridder";
static const std::string name_degridder = "kernel_degridder";
static const std::string name_calibrate = "kernel_calibrate";
static const std::string name_calibrate_hessian_vector_product1 =
    "kernel_calibrate_hessian_vector_product1";
static const std::string name_calibrate_hessian_vector_product2 =
    "kernel_calibrate_hessian_vector_product2";
static const std::string name_phasor = "kernel_phasor";
static const std::string name_adder = "kernel_adder";
static const std::string name_splitter = "kernel_splitter";
static const std::string name_adder_wstack = "kernel_adder_wstack";
static const std::string name_splitter_wstack = "kernel_splitter_wstack";
static const std::string name_adder_subgrids_to_wtiles =
    "kernel_adder_subgrids_to_wtiles";
static const std::string name_adder_wtiles_to_grid =
    "kernel_adder_wtiles_to_grid";
static const std::string name_splitter_subgrids_from_wtiles =
    "kernel_splitter_subgrids_from_wtiles";
static const std::string name_splitter_wtiles_from_grid =
    "kernel_splitter_wtiles_from_grid";
static const std::string name_tiles_to_grid = "kernel_tiles_to_grid";
static const std::string name_tiles_from_grid = "kernel_tiles_from_grid";
static const std::string name_fft = "kernel_fft";
static const std::string name_scaler = "kernel_scaler";
static const std::string name_average_beam = "kernel_average_beam";

}  // end namespace cpu
}  // end namespace kernel
}  // end namespace idg

#endif
