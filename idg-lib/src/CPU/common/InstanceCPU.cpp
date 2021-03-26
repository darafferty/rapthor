// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdint>   // unint64_t
#include <unistd.h>  // rmdir

#include <algorithm>  // transform

#include "idg-config.h"

#include "InstanceCPU.h"

using namespace std;

namespace idg {
namespace kernel {
namespace cpu {

// Constructor
InstanceCPU::InstanceCPU(std::vector<std::string> libraries)
    : KernelsInstance(),
      function_gridder(nullptr),
      function_degridder(nullptr),
      function_calibrate(nullptr),
      function_calibrate_hessian_vector_product1(nullptr),
      function_calibrate_hessian_vector_product2(nullptr),
      function_phasor(nullptr),
      function_fft(nullptr),
      function_adder(nullptr),
      function_splitter(nullptr),
      function_adder_wstack(nullptr),
      function_splitter_wstack(nullptr),
      function_adder_wtiles_to_grid(nullptr),
      function_splitter_wtiles_from_grid(nullptr),
      function_adder_subgrids_to_wtiles(nullptr),
      function_splitter_subgrids_from_wtiles(nullptr),
      function_average_beam(nullptr) {
#if defined(DEBUG)
  cout << __func__ << endl;
#endif

  m_powersensor.reset(powersensor::get_power_sensor(powersensor::sensor_host));

  load_shared_objects(libraries);
  load_kernel_funcions();
}

// Destructor
InstanceCPU::~InstanceCPU() {
#if defined(DEBUG)
  cout << __func__ << endl;
#endif

  // Unload shared objects by ~Module
  for (unsigned int i = 0; i < modules.size(); i++) {
    delete modules[i];
  }
}

void InstanceCPU::load_shared_objects(std::vector<std::string> libraries) {
#if defined(DEBUG)
  cout << __func__ << endl;
#endif

  for (auto library : libraries) {
    string library_ = auxiliary::get_lib_dir() + "/idg-cpu/" + library;

#if defined(DEBUG)
    cout << "Loading: " << library_ << endl;
#endif

    modules.push_back(new runtime::Module(library_.c_str()));
  }
}  // end load_shared_objects

// maps name -> index in modules that contain that symbol
void InstanceCPU::load_kernel_funcions() {
#if defined(DEBUG)
  cout << __func__ << endl;
#endif

  for (unsigned int i = 0; i < modules.size(); i++) {
    if (dlsym(*modules[i], kernel::cpu::name_gridder.c_str())) {
      function_gridder.reset(
          new runtime::Function(*modules[i], name_gridder.c_str()));
    }
    if (dlsym(*modules[i], kernel::cpu::name_degridder.c_str())) {
      function_degridder.reset(
          new runtime::Function(*modules[i], name_degridder.c_str()));
    }
    if (dlsym(*modules[i], kernel::cpu::name_calibrate.c_str())) {
      function_calibrate.reset(
          new runtime::Function(*modules[i], name_calibrate.c_str()));
    }
    if (dlsym(*modules[i],
              kernel::cpu::name_calibrate_hessian_vector_product1.c_str())) {
      function_calibrate_hessian_vector_product1.reset(new runtime::Function(
          *modules[i], name_calibrate_hessian_vector_product1.c_str()));
    }
    if (dlsym(*modules[i],
              kernel::cpu::name_calibrate_hessian_vector_product1.c_str())) {
      function_calibrate_hessian_vector_product2.reset(new runtime::Function(
          *modules[i], name_calibrate_hessian_vector_product2.c_str()));
    }
    if (dlsym(*modules[i], kernel::cpu::name_phasor.c_str())) {
      function_phasor.reset(
          new runtime::Function(*modules[i], name_phasor.c_str()));
    }
    if (dlsym(*modules[i], kernel::cpu::name_fft.c_str())) {
      function_fft.reset(new runtime::Function(*modules[i], name_fft.c_str()));
    }
    if (dlsym(*modules[i], kernel::cpu::name_adder.c_str())) {
      function_adder.reset(
          new runtime::Function(*modules[i], name_adder.c_str()));
    }
    if (dlsym(*modules[i], kernel::cpu::name_splitter.c_str())) {
      function_splitter.reset(
          new runtime::Function(*modules[i], name_splitter.c_str()));
    }
    if (dlsym(*modules[i], kernel::cpu::name_adder_wstack.c_str())) {
      function_adder_wstack.reset(
          new runtime::Function(*modules[i], name_adder_wstack.c_str()));
    }
    if (dlsym(*modules[i], kernel::cpu::name_splitter_wstack.c_str())) {
      function_splitter_wstack.reset(
          new runtime::Function(*modules[i], name_splitter_wstack.c_str()));
    }
    if (dlsym(*modules[i], kernel::cpu::name_adder_wtiles_to_grid.c_str())) {
      function_adder_wtiles_to_grid.reset(new runtime::Function(
          *modules[i], name_adder_wtiles_to_grid.c_str()));
    }
    if (dlsym(*modules[i],
              kernel::cpu::name_splitter_wtiles_from_grid.c_str())) {
      function_splitter_wtiles_from_grid.reset(new runtime::Function(
          *modules[i], name_splitter_wtiles_from_grid.c_str()));
    }
    if (dlsym(*modules[i],
              kernel::cpu::name_adder_subgrids_to_wtiles.c_str())) {
      function_adder_subgrids_to_wtiles.reset(new runtime::Function(
          *modules[i], name_adder_subgrids_to_wtiles.c_str()));
    }
    if (dlsym(*modules[i],
              kernel::cpu::name_splitter_subgrids_from_wtiles.c_str())) {
      function_splitter_subgrids_from_wtiles.reset(new runtime::Function(
          *modules[i], name_splitter_subgrids_from_wtiles.c_str()));
    }
    if (dlsym(*modules[i], kernel::cpu::name_average_beam.c_str())) {
      function_average_beam.reset(
          new runtime::Function(*modules[i], name_average_beam.c_str()));
    }
  }  // end for
}  // end load_kernel_funcions

// Function signatures
#define sig_gridder                                                       \
  (void (*)(int, int, int, float, float, const float *, int, int, void *, \
            void *, void *, void *, void *, void *, void *, void *, void *))
#define sig_degridder                                                     \
  (void (*)(int, int, int, float, float, const float *, int, int, void *, \
            void *, void *, void *, void *, void *, void *, void *))

#define sig_calibrate                                                        \
  (void (*)(const unsigned int nr_subgrids, const unsigned int grid_size,    \
            const unsigned int subgrid_size, const float image_size,         \
            const float w_step_in_lambda, const float *__restrict__ shift,   \
            const unsigned int max_nr_timesteps,                             \
            const unsigned int nr_channels, const unsigned int nr_stations,  \
            const unsigned int nr_terms, const unsigned int nr_time_slots,   \
            const idg::UVW<float> *uvw, const float *wavenumbers,            \
            idg::float2 *visibilities, const float *weights,                 \
            const idg::float2 *aterms, const idg::float2 *aterm_derivatives, \
            const int *aterms_indices, const idg::Metadata *metadata,        \
            const idg::float2 *subgrid, const idg::float2 *phasors,          \
            double *hessian, double *gradient, double *residual))

#define sig_calibrate_hessian_vector_product1                                \
  (void (*)(const unsigned int nr_subgrids, const unsigned int grid_size,    \
            const unsigned int subgrid_size, const float image_size,         \
            const float w_step_in_lambda, const float *__restrict__ shift,   \
            const unsigned int max_nr_timesteps,                             \
            const unsigned int nr_channels, const unsigned int nr_stations,  \
            const unsigned int nr_terms, const unsigned int nr_time_slots,   \
            const idg::UVW<float> *uvw, const float *wavenumbers,            \
            idg::float2 *visibilities, const float *weights,                 \
            const idg::float2 *aterms, const idg::float2 *aterm_derivatives, \
            const int *aterms_indices, const idg::Metadata *metadata,        \
            const idg::float2 *subgrid, const idg::float2 *phasors,          \
            const float *parameter_vector))

#define sig_calibrate_hessian_vector_product2 (void (*)())

#define sig_phasor                                                        \
  (void (*)(int, int, int, float, float, const float *, int, int, void *, \
            void *, void *, void *))
#define sig_fft (void (*)(long, long, long, void *, int))
#define sig_adder (void (*)(long, long, int, void *, void *, void *))
#define sig_splitter (void (*)(long, long, int, void *, void *, void *))
#define sig_adder_wstack (void (*)(long, long, int, void *, void *, void *))
#define sig_splitter_wstack (void (*)(long, long, int, void *, void *, void *))
#define sig_adder_subgrids_to_wtiles \
  (void (*)(long, int, int, int, void *, void *, void *))
#define sig_adder_wtiles_to_grid                                               \
  (void (*)(int grid_size, int subgrid_size, int wtile_size, float image_size, \
            float w_step, const float *shift, int nr_tiles, int *tile_ids,     \
            idg::Coordinate *tile_coordinates, idg::float2 *tiles,             \
            idg::float2 *grid))
#define sig_splitter_subgrids_from_wtiles \
  (void (*)(long, int, int, int, void *, void *, void *))
#define sig_splitter_wtiles_from_grid                                          \
  (void (*)(int grid_size, int subgrid_size, int wtile_size, float image_size, \
            float w_step, const float *shift, int nr_tiles, int *tile_ids,     \
            idg::Coordinate *tile_coordinates, idg::float2 *tiles,             \
            idg::float2 *grid))

#define sig_average_beam                                                  \
  (void (*)(int, int, int, int, int, int, void *, void *, void *, void *, \
            void *, void *))

void InstanceCPU::run_gridder(int nr_subgrids, int grid_size, int subgrid_size,
                              float image_size, float w_step,
                              const float *shift, int nr_channels,
                              int nr_stations, void *uvw, void *wavenumbers,
                              void *visibilities, void *spheroidal, void *aterm,
                              void *aterm_idx, void *avg_aterm, void *metadata,
                              void *subgrid) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();
  (sig_gridder(void *) * function_gridder)(
      nr_subgrids, grid_size, subgrid_size, image_size, w_step, shift,
      nr_channels, nr_stations, uvw, wavenumbers, visibilities, spheroidal,
      aterm, aterm_idx, avg_aterm, metadata, subgrid);
  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::gridder, states[0], states[1]);
  }
}

void InstanceCPU::run_degridder(int nr_subgrids, int grid_size,
                                int subgrid_size, float image_size,
                                float w_step, const float *shift,
                                int nr_channels, int nr_stations, void *uvw,
                                void *wavenumbers, void *visibilities,
                                void *spheroidal, void *aterm, void *aterm_idx,
                                void *metadata, void *subgrid) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();
  (sig_degridder(void *) * function_degridder)(
      nr_subgrids, grid_size, subgrid_size, image_size, w_step, shift,
      nr_channels, nr_stations, uvw, wavenumbers, visibilities, spheroidal,
      aterm, aterm_idx, metadata, subgrid);
  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::degridder, states[0], states[1]);
  }
}

void InstanceCPU::run_average_beam(int nr_baselines, int nr_antennas,
                                   int nr_timesteps, int nr_channels,
                                   int nr_aterms, int subgrid_size, void *uvw,
                                   void *baselines, void *aterms,
                                   void *aterms_offsets, void *weights,
                                   void *average_beam) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();
  (sig_average_beam(void *) * function_average_beam)(
      nr_baselines, nr_antennas, nr_timesteps, nr_channels, nr_aterms,
      subgrid_size, uvw, baselines, aterms, aterms_offsets, weights,
      average_beam);
  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::average_beam, states[0], states[1]);
  }
}

void InstanceCPU::run_calibrate(
    int nr_subgrids, int grid_size, int subgrid_size, float image_size,
    float w_step, const float *shift, int max_nr_timesteps, int nr_channels,
    int nr_terms, int nr_stations, int nr_time_slots,
    const idg::UVW<float> *uvw, const float *wavenumbers,
    idg::float2 *visibilities, const float *weights, const idg::float2 *aterm,
    const idg::float2 *aterm_derivative, const int *aterms_indices,
    const idg::Metadata *metadata, const idg::float2 *subgrid,
    const idg::float2 *phasors, double *hessian, double *gradient,
    double *residual) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();
  (sig_calibrate(void *) * function_calibrate)(
      nr_subgrids, grid_size, subgrid_size, image_size, w_step, shift,
      max_nr_timesteps, nr_channels, nr_stations, nr_terms, nr_time_slots, uvw,
      wavenumbers, visibilities, weights, aterm, aterm_derivative,
      aterms_indices, metadata, subgrid, phasors, hessian, gradient, residual);
  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update<Report::calibrate>(states[0], states[1]);
  }
}

void InstanceCPU::run_calibrate_hessian_vector_product1(
    const int station_nr, const Array4D<Matrix2x2<std::complex<float>>> &aterms,
    const Array4D<Matrix2x2<std::complex<float>>> &derivative_aterms,
    const Array2D<float> &parameter_vector) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();
  //                 (sig_calibrate_hessian_vector_product1 (void *)
  //                 *function_calibrate_hessian_vector_product1)();
  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::calibrate, states[0], states[1]);
  }
}

void InstanceCPU::run_calibrate_hessian_vector_product2(
    const int station_nr, const Array4D<Matrix2x2<std::complex<float>>> &aterms,
    const Array4D<Matrix2x2<std::complex<float>>> &derivative_aterms,
    Array2D<float> &parameter_vector) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();
  (sig_calibrate_hessian_vector_product2(void *) *
   function_calibrate_hessian_vector_product2)();
  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::calibrate, states[0], states[1]);
  }
}

void InstanceCPU::run_phasor(int nr_subgrids, int grid_size, int subgrid_size,
                             float image_size, float w_step, const float *shift,
                             int max_nr_timesteps, int nr_channels, void *uvw,
                             void *wavenumbers, void *metadata, void *phasors) {
  (sig_phasor(void *) * function_phasor)(
      nr_subgrids, grid_size, subgrid_size, image_size, w_step, shift,
      max_nr_timesteps, nr_channels, uvw, wavenumbers, metadata, phasors);
}

void InstanceCPU::run_fft(int grid_size, int size, int batch, void *data,
                          int direction) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();
  (sig_fft(void *) * function_fft)(grid_size, size, batch, data, direction);
  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::grid_fft, states[0], states[1]);
  }
}

void InstanceCPU::run_subgrid_fft(int grid_size, int size, int batch,
                                  void *data, int direction) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();
  (sig_fft(void *) * function_fft)(grid_size, size, batch, data, direction);
  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::subgrid_fft, states[0], states[1]);
  }
}

void InstanceCPU::run_adder(int nr_subgrids, int grid_size, int subgrid_size,
                            void *metadata, void *subgrid, void *grid) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();
  (sig_adder(void *) * function_adder)(nr_subgrids, grid_size, subgrid_size,
                                       metadata, subgrid, grid);
  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::adder, states[0], states[1]);
  }
}

void InstanceCPU::run_splitter(int nr_subgrids, int grid_size, int subgrid_size,
                               void *metadata, void *subgrid, void *grid) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();
  (sig_splitter(void *) * function_splitter)(
      nr_subgrids, grid_size, subgrid_size, metadata, subgrid, grid);
  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::splitter, states[0], states[1]);
  }
}

void InstanceCPU::run_adder_wstack(int nr_subgrids, int grid_size,
                                   int subgrid_size, void *metadata,
                                   void *subgrid, void *grid) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();
  (sig_adder_wstack(void *) * function_adder_wstack)(
      nr_subgrids, grid_size, subgrid_size, metadata, subgrid, grid);
  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::adder, states[0], states[1]);
  }
}

void InstanceCPU::run_splitter_wstack(int nr_subgrids, int grid_size,
                                      int subgrid_size, void *metadata,
                                      void *subgrid, void *grid) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();
  (sig_splitter_wstack(void *) * function_splitter_wstack)(
      nr_subgrids, grid_size, subgrid_size, metadata, subgrid, grid);
  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::splitter, states[0], states[1]);
  }
}

void InstanceCPU::run_adder_subgrids_to_wtiles(int nr_subgrids, int grid_size,
                                               int subgrid_size, void *metadata,
                                               void *subgrid) {
  (sig_adder_subgrids_to_wtiles(void *) * function_adder_subgrids_to_wtiles)(
      nr_subgrids, grid_size, subgrid_size, kWTileSize, metadata, subgrid,
      m_wtiles_buffer.data());
}

void InstanceCPU::run_splitter_subgrids_from_wtiles(int nr_subgrids,
                                                    int grid_size,
                                                    int subgrid_size,
                                                    void *metadata,
                                                    void *subgrid) {
  (sig_splitter_subgrids_from_wtiles(void *) *
   function_splitter_subgrids_from_wtiles)(nr_subgrids, grid_size, subgrid_size,
                                           kWTileSize, metadata, subgrid,
                                           m_wtiles_buffer.data());
}

void InstanceCPU::run_adder_wtiles_to_grid(int grid_size, int subgrid_size,
                                           float image_size, float w_step,
                                           const float *shift, int nr_tiles,
                                           int *tile_ids,
                                           idg::Coordinate *tile_coordinates,
                                           std::complex<float> *grid) {
  (sig_adder_wtiles_to_grid(void *) * function_adder_wtiles_to_grid)(
      grid_size, subgrid_size, kWTileSize, image_size, w_step, shift, nr_tiles,
      tile_ids, tile_coordinates,
      reinterpret_cast<idg::float2 *>(m_wtiles_buffer.data()),
      reinterpret_cast<idg::float2 *>(grid));
}

void InstanceCPU::run_splitter_wtiles_from_grid(int grid_size, int subgrid_size,
                                                float image_size, float w_step,
                                                const float *shift,
                                                int nr_tiles, int *tile_ids,
                                                Coordinate *tile_coordinates,
                                                std::complex<float> *grid) {
  (sig_splitter_wtiles_from_grid(void *) * function_splitter_wtiles_from_grid)(
      grid_size, subgrid_size, kWTileSize, image_size, w_step, shift, nr_tiles,
      tile_ids, tile_coordinates,
      reinterpret_cast<float2 *>(m_wtiles_buffer.data()),
      reinterpret_cast<float2 *>(grid));
}

void InstanceCPU::run_adder_wtiles(
    unsigned int nr_subgrids, unsigned int grid_size, unsigned int subgrid_size,
    float image_size, float w_step, const float *shift, int subgrid_offset,
    WTileUpdateSet &wtile_flush_set, void *metadata, void *subgrid,
    std::complex<float> *grid) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();

  for (int subgrid_index = 0; subgrid_index < (int)nr_subgrids;) {
    // Is a flush needed right now?
    if (!wtile_flush_set.empty() && wtile_flush_set.front().subgrid_index ==
                                        subgrid_index + subgrid_offset) {
      // Get information on what wtiles to flush
      WTileUpdateInfo &wtile_flush_info = wtile_flush_set.front();
      // Project wtiles to master grid
      run_adder_wtiles_to_grid(grid_size, subgrid_size, image_size, w_step,
                               shift, wtile_flush_info.wtile_ids.size(),
                               wtile_flush_info.wtile_ids.data(),
                               wtile_flush_info.wtile_coordinates.data(), grid);
      // Remove the flush event from the queue
      wtile_flush_set.pop_front();
    }

    // Initialize number of subgrids to process next to all remaining subgrids
    // in job
    int nr_subgrids_to_process = nr_subgrids - subgrid_index;

    // Check whether a flush needs to happen before the end of the job
    if (!wtile_flush_set.empty() && wtile_flush_set.front().subgrid_index -
                                            (subgrid_index + subgrid_offset) <
                                        nr_subgrids_to_process) {
      // Reduce the number of subgrids to process to just before the next flush
      // event
      nr_subgrids_to_process = wtile_flush_set.front().subgrid_index -
                               (subgrid_index + subgrid_offset);
    }

    // Add all subgrids than can be added to the wtiles
    run_adder_subgrids_to_wtiles(
        nr_subgrids_to_process, grid_size, subgrid_size,
        &static_cast<Metadata *>(metadata)[subgrid_index],
        &static_cast<std::complex<float> *>(
            subgrid)[subgrid_index * subgrid_size * subgrid_size *
                     NR_CORRELATIONS]);
    // Increment the subgrid index by the actual number of processed subgrids
    subgrid_index += nr_subgrids_to_process;
  }

  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::adder, states[0], states[1]);
  }
}

void InstanceCPU::run_splitter_wtiles(int nr_subgrids, int grid_size,
                                      int subgrid_size, float image_size,
                                      float w_step, const float *shift,
                                      int subgrid_offset,
                                      WTileUpdateSet &wtile_initialize_set,
                                      void *metadata, void *subgrid,
                                      std::complex<float> *grid) {
  powersensor::State states[2];
  states[0] = m_powersensor->read();

  for (int subgrid_index = 0; subgrid_index < nr_subgrids;) {
    // Check whether initialize is needed right now
    if (!wtile_initialize_set.empty() &&
        wtile_initialize_set.front().subgrid_index ==
            (int)(subgrid_index + subgrid_offset)) {
      // Get the information on what wtiles to initialize
      WTileUpdateInfo &wtile_initialize_info = wtile_initialize_set.front();
      // Initialize the wtiles from the grid
      run_splitter_wtiles_from_grid(
          grid_size, subgrid_size, image_size, w_step, shift,
          wtile_initialize_info.wtile_ids.size(),
          wtile_initialize_info.wtile_ids.data(),
          wtile_initialize_info.wtile_coordinates.data(), grid);
      // Remove initialize even from queue
      wtile_initialize_set.pop_front();
    }

    // Initialize number of subgrids to proccess next to all remaining subgrids
    // in job
    int nr_subgrids_to_process = nr_subgrids - subgrid_index;

    // Check whether initialization needs to happen before the end of the job
    if (!wtile_initialize_set.empty() &&
        wtile_initialize_set.front().subgrid_index -
                (subgrid_index + subgrid_offset) <
            nr_subgrids_to_process) {
      // Reduce the number of subgrids to process to just before the next
      // initialization event
      nr_subgrids_to_process = wtile_initialize_set.front().subgrid_index -
                               (subgrid_offset + subgrid_index);
    }

    // Process all subgrids that can be processed now
    run_splitter_subgrids_from_wtiles(
        nr_subgrids_to_process, grid_size, subgrid_size,
        &static_cast<Metadata *>(metadata)[subgrid_index],
        &static_cast<std::complex<float> *>(
            subgrid)[subgrid_index * subgrid_size * subgrid_size *
                     NR_CORRELATIONS]);

    // Increment the subgrid index by the actual number of processed subgrids
    subgrid_index += nr_subgrids_to_process;
  }  // end for subgrid_index

  states[1] = m_powersensor->read();
  if (m_report) {
    m_report->update(Report::splitter, states[0], states[1]);
  }
}  // end run_splitter_wtiles

size_t InstanceCPU::init_wtiles(size_t grid_size, int subgrid_size) {
  // Heuristic for choosing the number of wtiles.
  // A number that is too small will result in excessive flushing, too large in
  // excessive memory usage.
  //
  // Current heuristic is for the wtiles to cover 50% the grid.
  // Because of padding with subgrid_size, the memory used will be more than 50%
  // of the memory used for the grid. In the extreme case subgrid_size is
  // equal to kWTileSize (both 128) m_wtiles_buffer will be as large as the
  // grid. The minimum number of wtiles is 4.
  size_t nr_wtiles_min = 4;
  size_t nr_wtiles = std::max(
      nr_wtiles_min, (grid_size * grid_size) / (kWTileSize * kWTileSize) / 2);

  // Make sure that the wtiles buffer does not use an excessive amount of memory
  const size_t padded_wtile_size = size_t(kWTileSize) + size_t(subgrid_size);
  size_t sizeof_padded_wtile = NR_CORRELATIONS * padded_wtile_size *
                               padded_wtile_size * sizeof(std::complex<float>);
  size_t sizeof_padded_wtiles = nr_wtiles * sizeof_padded_wtile;
  size_t free_memory = auxiliary::get_free_memory() * 1024 * 1024;  // Bytes
  while (sizeof_padded_wtiles > free_memory) {
    nr_wtiles *= 0.9;
    sizeof_padded_wtiles = nr_wtiles * sizeof_padded_wtiles;
  }
  assert(nr_wtiles >= nr_wtiles_min);

  m_wtiles_buffer = idg::Array1D<std::complex<float>>(
      sizeof_padded_wtiles / sizeof(std::complex<float>));
  m_wtiles_buffer.zero();
  return nr_wtiles;
}

}  // namespace cpu
}  // namespace kernel
}  // namespace idg
