// Copyright (C) 2023 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <iomanip>  // setprecision

#include <cudawrappers/cu.hpp>

#include "InstanceCUDA.h"
#include "PowerRecord.h"

namespace idg::kernel::cuda {

// Constructor
InstanceCUDA::InstanceCUDA(size_t device_id) : KernelsInstance() {
#if defined(DEBUG)
  std::cout << __func__ << std::endl;
#endif

  // Initialize members
  device_.reset(new cu::Device(device_id));
  context_.reset(new cu::Context(CU_CTX_SCHED_BLOCKING_SYNC, *device_));
  stream_execute_.reset(new cu::Stream());
  stream_htod_.reset(new cu::Stream());
  stream_dtoh_.reset(new cu::Stream());
  report_ = std::make_shared<Report>();
  power_meter_ = pmt::get_power_meter(pmt::sensor_device, device_id);
}

// Destructor
InstanceCUDA::~InstanceCUDA() {
#if defined(DEBUG)
  std::cout << __func__ << std::endl;
#endif

  free_events();
  stream_execute_.reset();
  stream_htod_.reset();
  stream_dtoh_.reset();
  context_.reset();
  device_.reset();
}

std::ostream& operator<<(std::ostream& os, InstanceCUDA& d) {
  os << d.get_device().getName() << std::endl;
  os << std::setprecision(2);
  os << std::fixed;

  // Device memory
  const size_t device_memory_total =
      d.get_context().getTotalMemory() / (1024 * 1024);  // MBytes
  const size_t device_memory_free =
      d.get_context().getFreeMemory() / (1024 * 1024);  // MBytes
  os << "\tDevice memory : " << device_memory_free << " Mb  / "
     << device_memory_total << " Mb (free / total)" << std::endl;

  // Shared memory
  const int shared_memory = d.get_device().getAttribute(
      CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);  // Bytes
  os << "\tShared memory : " << shared_memory / (float)1024 << " Kb"
     << std::endl;

  // Frequencies
  const int clock_frequency =
      d.get_device().getAttribute(CU_DEVICE_ATTRIBUTE_CLOCK_RATE) /
      1000;  // Mhz
  os << "\tClk frequency : " << clock_frequency << " Ghz" << std::endl;
  const int mem_frequency =
      d.get_device().getAttribute(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE) /
      1000;  // Mhz
  os << "\tMem frequency : " << mem_frequency << " Ghz" << std::endl;

  // Cores/bus
  const int nr_sm =
      d.get_device().getAttribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
  const int mem_bus_width = d.get_device().getAttribute(
      CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH);  // Bits
  os << "\tNumber of SM  : " << nr_sm << std::endl;
  os << "\tMem bus width : " << mem_bus_width << " bit" << std::endl;
  os << "\tMem bandwidth : " << 2 * (mem_bus_width / 8) * mem_frequency / 1000
     << " GB/s" << std::endl;

  const int nr_threads =
      d.get_device()
          .getAttribute<CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR>();
  os << "\tNumber of threads  : " << nr_threads << std::endl;

  // Misc
  const int capability =
      10 * d.get_device().getAttribute(
               CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) +
      d.get_device().getAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
  os << "\tCapability    : " << capability << std::endl;

  // Unified memory
  const bool supports_managed_memory =
      d.get_device().getAttribute(CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY);
  os << "\tUnified memory : " << supports_managed_memory << std::endl;

  os << std::endl;
  return os;
}

pmt::State InstanceCUDA::measure() { return power_meter_->Read(); }

void InstanceCUDA::measure(PowerRecord& record, cu::Stream& stream) {
  record.sensor = *power_meter_;
  record.enqueue(stream);
}

cu::Event& InstanceCUDA::get_event() {
  // Create new event
  cu::Event* event = new cu::Event();

  // This event is used in a callback, where it can not be destroyed
  // after use. Instead, register the event globally, and take care of
  // destruction there.
  events.emplace_back(event);

  // Return a reference to the event
  return *event;
}

typedef struct {
  PowerRecord* start;
  PowerRecord* end;
  std::shared_ptr<Report> report;
  Report::ID id;
  void (Report::*update_report)(pmt::State&, pmt::State&);
} UpdateData;

UpdateData* get_update_data(cu::Event& event, pmt::Pmt& sensor,
                            std::shared_ptr<Report> report, Report::ID id) {
  UpdateData* data = new UpdateData();
  data->start = new PowerRecord(event, sensor);
  data->end = new PowerRecord(event, sensor);
  data->report = report;
  data->id = id;
  return data;
}

void update_report_callback(CUstream, CUresult, void* userData) {
  UpdateData* data = static_cast<UpdateData*>(userData);
  PowerRecord* start = data->start;
  PowerRecord* end = data->end;
  data->report->update(data->id, start->state, end->state);
  delete start;
  delete end;
  delete data;
}

void InstanceCUDA::start_measurement(void* ptr) {
  UpdateData* data = (UpdateData*)ptr;

  // Schedule the first measurement (prior to kernel execution)
  data->start->enqueue(*stream_execute_);
}

void InstanceCUDA::end_measurement(void* ptr) {
  UpdateData* data = (UpdateData*)ptr;

  // Schedule the second measurement (after the kernel execution)
  data->end->enqueue(*stream_execute_);

  // Afterwards, update the report according to the two measurements
  stream_execute_->addCallback(
      static_cast<CUstreamCallback>(&update_report_callback), data);
}

void InstanceCUDA::launch_gridder(
    int time_offset, int nr_subgrids, int nr_polarizations, int grid_size,
    int subgrid_size, float image_size, float w_step, int nr_channels,
    int nr_stations, float shift_l, float shift_m, cu::DeviceMemory& d_uvw,
    cu::DeviceMemory& d_wavenumbers, cu::DeviceMemory& d_visibilities,
    cu::DeviceMemory& d_taper, cu::DeviceMemory& d_aterms,
    cu::DeviceMemory& d_aterm_indices, cu::DeviceMemory& d_metadata,
    cu::DeviceMemory& d_avg_aterm, cu::DeviceMemory& d_subgrid) {
  const bool enable_avg_aterm = d_avg_aterm.size();
  compile_gridder(enable_avg_aterm);
  UpdateData* data =
      get_update_data(get_event(), *power_meter_, report_, Report::gridder);
  start_measurement(data);
  kernel_gridder_->enqueue(time_offset, nr_subgrids, nr_polarizations,
                           grid_size, subgrid_size, image_size, w_step,
                           nr_channels, nr_stations, shift_l, shift_m, d_uvw,
                           d_wavenumbers, d_visibilities, d_taper, d_aterms,
                           d_aterm_indices, d_metadata, d_avg_aterm, d_subgrid);
  end_measurement(data);
}

void InstanceCUDA::launch_degridder(
    int time_offset, int nr_subgrids, int nr_polarizations, int grid_size,
    int subgrid_size, float image_size, float w_step, int nr_channels,
    int nr_stations, float shift_l, float shift_m, cu::DeviceMemory& d_uvw,
    cu::DeviceMemory& d_wavenumbers, cu::DeviceMemory& d_visibilities,
    cu::DeviceMemory& d_taper, cu::DeviceMemory& d_aterms,
    cu::DeviceMemory& d_aterm_indices, cu::DeviceMemory& d_metadata,
    cu::DeviceMemory& d_subgrid) {
  compile_degridder();
  UpdateData* data =
      get_update_data(get_event(), *power_meter_, report_, Report::degridder);
  start_measurement(data);
  kernel_degridder_->enqueue(time_offset, nr_subgrids, nr_polarizations,
                             grid_size, subgrid_size, image_size, w_step,
                             nr_channels, nr_stations, shift_l, shift_m, d_uvw,
                             d_wavenumbers, d_visibilities, d_taper, d_aterms,
                             d_aterm_indices, d_metadata, d_subgrid);
  end_measurement(data);
}

void InstanceCUDA::launch_average_beam(
    int nr_baselines, int nr_antennas, int nr_timesteps, int nr_channels,
    int nr_aterms, int subgrid_size, cu::DeviceMemory& d_uvw,
    cu::DeviceMemory& d_baselines, cu::DeviceMemory& d_aterms,
    cu::DeviceMemory& d_aterm_offsets, cu::DeviceMemory& d_weights,
    cu::DeviceMemory& d_average_beam) {
  compile_average_beam();
  UpdateData* data = get_update_data(get_event(), *power_meter_, report_,
                                     Report::average_beam);
  start_measurement(data);
  kernel_average_beam_->enqueue(nr_baselines, nr_antennas, nr_timesteps,
                                nr_channels, nr_aterms, subgrid_size, d_uvw,
                                d_baselines, d_aterms, d_aterm_offsets,
                                d_weights, d_average_beam);
  end_measurement(data);
}

void InstanceCUDA::launch_calibrate(
    int nr_subgrids, int grid_size, int subgrid_size, float image_size,
    float w_step, int total_nr_timesteps, int nr_channels, int nr_stations,
    int nr_terms, cu::DeviceMemory& d_uvw, cu::DeviceMemory& d_wavenumbers,
    cu::DeviceMemory& d_visibilities, cu::DeviceMemory& d_weights,
    cu::DeviceMemory& d_aterm, cu::DeviceMemory& d_aterm_derivatives,
    cu::DeviceMemory& d_aterm_indices, cu::DeviceMemory& d_metadata,
    cu::DeviceMemory& d_subgrid, cu::DeviceMemory& d_sums1,
    cu::DeviceMemory& d_sums2, cu::DeviceMemory& d_lmnp,
    cu::DeviceMemory& d_hessian, cu::DeviceMemory& d_gradient,
    cu::DeviceMemory& d_residual) {
  compile_calibrate();
  UpdateData* data =
      get_update_data(get_event(), *power_meter_, report_, Report::calibrate);
  start_measurement(data);

  // Precompute l,m,n and phase offset
  kernel_calibrate_lmnp_->enqueue(nr_subgrids, grid_size, subgrid_size,
                                  image_size, w_step, d_metadata, d_lmnp);

  const unsigned int nr_polarizations = 4;
  const unsigned int max_nr_terms = 8;
  unsigned int current_nr_terms_y = max_nr_terms;
  for (unsigned int term_offset_y = 0; term_offset_y < (unsigned int)nr_terms;
       term_offset_y += current_nr_terms_y) {
    const unsigned int last_term_y =
        min(nr_terms, term_offset_y + current_nr_terms_y);
    unsigned int current_nr_terms_y = last_term_y - term_offset_y;

    // Compute sums1
    kernel_calibrate_sums_->enqueue(
        nr_subgrids, nr_polarizations, subgrid_size, image_size,
        total_nr_timesteps, nr_channels, nr_stations, term_offset_y,
        current_nr_terms_y, nr_terms, d_uvw, d_wavenumbers, d_aterm,
        d_aterm_derivatives, d_aterm_indices, d_metadata, d_subgrid, d_sums1,
        d_lmnp);

    // Compute gradient (diagonal)
    if (term_offset_y == 0) {
      kernel_calibrate_gradient_->enqueue(
          nr_subgrids, nr_polarizations, subgrid_size, image_size,
          total_nr_timesteps, nr_channels, nr_stations, term_offset_y,
          current_nr_terms_y, nr_terms, d_uvw, d_wavenumbers, d_visibilities,
          d_weights, d_aterm, d_aterm_derivatives, d_aterm_indices, d_metadata,
          d_subgrid, d_sums1, d_lmnp, d_gradient, d_residual);
    }

    // Compute hessian (diagonal)
    kernel_calibrate_hessian_->enqueue(
        nr_subgrids, current_nr_terms_y, current_nr_terms_y, nr_polarizations,
        total_nr_timesteps, nr_channels, term_offset_y, term_offset_y, nr_terms,
        d_weights, d_aterm_indices, d_metadata, d_sums1, d_sums1, d_hessian);

    unsigned int current_nr_terms_x = max_nr_terms;
    for (unsigned int term_offset_x = last_term_y;
         term_offset_x < (unsigned int)nr_terms;
         term_offset_x += current_nr_terms_x) {
      unsigned int last_term_x =
          min(nr_terms, term_offset_x + current_nr_terms_x);
      current_nr_terms_x = last_term_x - term_offset_x;

      // Compute sums2 (horizontal offset)
      kernel_calibrate_sums_->enqueue(
          nr_subgrids, nr_polarizations, subgrid_size, image_size,
          total_nr_timesteps, nr_channels, nr_stations, term_offset_x,
          current_nr_terms_x, nr_terms, d_uvw, d_wavenumbers, d_aterm,
          d_aterm_derivatives, d_aterm_indices, d_metadata, d_subgrid, d_sums2,
          d_lmnp);

      // Compute gradient (horizontal offset)
      if (term_offset_y == 0) {
        kernel_calibrate_gradient_->enqueue(
            nr_subgrids, nr_polarizations, subgrid_size, image_size,
            total_nr_timesteps, nr_channels, nr_stations, term_offset_x,
            current_nr_terms_x, nr_terms, d_uvw, d_wavenumbers, d_visibilities,
            d_weights, d_aterm, d_aterm_derivatives, d_aterm_indices,
            d_metadata, d_subgrid, d_sums2, d_lmnp, d_gradient, d_residual);
      }

      // Compute hessian (horizontal offset)
      kernel_calibrate_hessian_->enqueue(
          nr_subgrids, current_nr_terms_x, current_nr_terms_y, nr_polarizations,
          total_nr_timesteps, nr_channels, term_offset_y, term_offset_x,
          nr_terms, d_weights, d_aterm_indices, d_metadata, d_sums1, d_sums2,
          d_hessian);
    }
  }
  end_measurement(data);
}

void InstanceCUDA::launch_grid_fft(cu::DeviceMemory& d_data, int batch,
                                   long grid_size, DomainAtoDomainB direction) {
  // Plan FFT
  std::unique_ptr<KernelFFT> kernel =
      KernelFFT::create(*stream_execute_, {size_t(grid_size), size_t(batch)});

  // Enqueue start of measurement
  UpdateData* data =
      get_update_data(get_event(), *power_meter_, report_, Report::grid_fft);
  start_measurement(data);

  // Enqueue fft for the entire batch
  kernel->enqueue(d_data, direction, batch);

  // Enqueue end of measurement
  end_measurement(data);
}

std::unique_ptr<KernelFFT> InstanceCUDA::plan_batched_fft(size_t size,
                                                          size_t batch) {
  // Amount of device memory free (with a small safety margin)
  const size_t bytes_free = get_free_memory() * 0.95;

  // Amount of device memory required for temporary buffer and the FFT
  // plan
  const size_t bytes_required =
      2 * batch * size * size * sizeof(std::complex<float>);

  // Compute the actual batch size to use
  if (bytes_required > bytes_free) {
    batch *= (bytes_free / bytes_required);
  }
  batch = std::min(kFftBatch, batch);

  // Plan fft
  return KernelFFT::create(*stream_execute_, {size, batch});
}

void InstanceCUDA::launch_batched_fft(KernelFFT& kernel,
                                      cu::DeviceMemory& d_data, size_t batch,
                                      DomainAtoDomainB direction) {
  // Enqueue start of measurement
  UpdateData* data =
      get_update_data(get_event(), *power_meter_, report_, Report::subgrid_fft);
  start_measurement(data);

  // Enqeueue fft
  kernel.enqueue(d_data, direction, batch);

  // Enqueue end of measurement
  end_measurement(data);
}

void InstanceCUDA::launch_fft_shift(cu::DeviceMemory& d_data, int batch,
                                    long size, std::complex<float> scale) {
  compile_fft_shift();
  UpdateData* data =
      get_update_data(get_event(), *power_meter_, report_, Report::fft_shift);
  start_measurement(data);
  kernel_fft_shift_->enqueue(d_data, size, batch, scale);
  end_measurement(data);
}

void InstanceCUDA::launch_adder(int nr_subgrids, int nr_polarizations,
                                long grid_size, int subgrid_size,
                                cu::DeviceMemory& d_metadata,
                                cu::DeviceMemory& d_subgrid,
                                cu::DeviceMemory& d_grid) {
  compile_adder();
  UpdateData* data =
      get_update_data(get_event(), *power_meter_, report_, Report::adder);
  start_measurement(data);
  kernel_adder_->enqueue(nr_subgrids, nr_polarizations, grid_size, subgrid_size,
                         d_metadata, d_subgrid, d_grid);
  end_measurement(data);
}

void InstanceCUDA::launch_splitter(int nr_subgrids, int nr_polarizations,
                                   long grid_size, int subgrid_size,
                                   cu::DeviceMemory& d_metadata,
                                   cu::DeviceMemory& d_subgrid,
                                   cu::DeviceMemory& d_grid) {
  compile_splitter();
  UpdateData* data =
      get_update_data(get_event(), *power_meter_, report_, Report::splitter);
  start_measurement(data);
  kernel_splitter_->enqueue(nr_subgrids, nr_polarizations, grid_size,
                            subgrid_size, d_metadata, d_subgrid, d_grid);
  end_measurement(data);
}

void InstanceCUDA::launch_scaler(int nr_subgrids, int nr_polarizations,
                                 int subgrid_size,
                                 cu::DeviceMemory& d_subgrid) {
  compile_scaler();
  UpdateData* data =
      get_update_data(get_event(), *power_meter_, report_, Report::fft_scale);
  start_measurement(data);
  kernel_scaler_->enqueue(nr_subgrids, nr_polarizations, subgrid_size,
                          d_subgrid);
  end_measurement(data);
}

void InstanceCUDA::launch_copy_tiles(
    unsigned int nr_polarizations, unsigned int nr_tiles,
    unsigned int src_tile_size, unsigned int dst_tile_size,
    cu::DeviceMemory& d_src_tile_ids, cu::DeviceMemory& d_dst_tile_ids,
    cu::DeviceMemory& d_src_tiles, cu::DeviceMemory& d_dst_tiles) {
  compile_wtiling();
  kernel_wtiling_copy_->enqueue(nr_polarizations, nr_tiles, src_tile_size,
                                dst_tile_size, d_src_tile_ids, d_dst_tile_ids,
                                d_src_tiles, d_dst_tiles);
}

void InstanceCUDA::launch_apply_phasor_to_wtiles(
    unsigned int nr_polarizations, unsigned int nr_tiles, float image_size,
    float w_step, unsigned int tile_size, cu::DeviceMemory& d_tiles,
    cu::DeviceMemory& d_shift, cu::DeviceMemory& d_tile_coordinates, int sign) {
  compile_wtiling();
  kernel_wtiling_phasor_->enqueue(nr_polarizations, nr_tiles, image_size,
                                  w_step, tile_size, d_tiles, d_shift,
                                  d_tile_coordinates, sign);
}

void InstanceCUDA::launch_adder_subgrids_to_wtiles(
    int nr_subgrids, int nr_polarizations, long grid_size, int subgrid_size,
    int tile_size, int subgrid_offset, cu::DeviceMemory& d_metadata,
    cu::DeviceMemory& d_subgrid, cu::DeviceMemory& d_tiles,
    std::complex<float> scale) {
  compile_wtiling();
  kernel_wtiling_subgrids_to_wtiles_->enqueue(
      nr_subgrids, nr_polarizations, grid_size, subgrid_size, tile_size,
      subgrid_offset, d_metadata, d_subgrid, d_tiles, scale);
}

void InstanceCUDA::launch_adder_wtiles_to_grid(
    int nr_polarizations, int nr_tiles, long grid_size, int tile_size,
    int padded_tile_size, cu::DeviceMemory& d_tile_ids,
    cu::DeviceMemory& d_tile_coordinates, cu::DeviceMemory& d_tiles,
    cu::DeviceMemory& d_grid) {
  compile_wtiling();
  kernel_wtiling_wtiles_to_grid_->enqueue(
      nr_polarizations, nr_tiles, grid_size, tile_size, padded_tile_size,
      d_tile_ids, d_tile_coordinates, d_tiles, d_grid);
}

void InstanceCUDA::launch_splitter_subgrids_from_wtiles(
    int nr_subgrids, int nr_polarizations, long grid_size, int subgrid_size,
    int tile_size, int subgrid_offset, cu::DeviceMemory& d_metadata,
    cu::DeviceMemory& d_subgrid, cu::DeviceMemory& d_tiles) {
  compile_wtiling();
  kernel_wtiling_subgrids_from_wtiles_->enqueue(
      nr_subgrids, nr_polarizations, grid_size, subgrid_size, tile_size,
      subgrid_offset, d_metadata, d_subgrid, d_tiles);
}

void InstanceCUDA::launch_splitter_wtiles_from_grid(
    int nr_polarizations, int nr_tiles, long grid_size, int tile_size,
    int padded_tile_size, cu::DeviceMemory& d_tile_ids,
    cu::DeviceMemory& d_tile_coordinates, cu::DeviceMemory& d_tiles,
    cu::DeviceMemory& d_grid) {
  compile_wtiling();
  kernel_wtiling_wtiles_from_grid_->enqueue(
      nr_polarizations, nr_tiles, grid_size, tile_size, padded_tile_size,
      d_tile_ids, d_tile_coordinates, d_tiles, d_grid);
}

void InstanceCUDA::launch_adder_wtiles_to_patch(
    int nr_polarizations, int nr_tiles, long grid_size, int tile_size,
    int padded_tile_size, int patch_size, idg::Coordinate patch_coordinate,
    cu::DeviceMemory& d_tile_ids, cu::DeviceMemory& d_tile_coordinates,
    cu::DeviceMemory& d_tiles, cu::DeviceMemory& d_patch) {
  compile_wtiling();
  kernel_wtiling_wtiles_to_patch_->enqueue(
      nr_polarizations, nr_tiles, grid_size, tile_size, padded_tile_size,
      patch_size, patch_coordinate, d_tile_ids, d_tile_coordinates, d_tiles,
      d_patch);
}

void InstanceCUDA::launch_splitter_wtiles_from_patch(
    int nr_polarizations, int nr_tiles, long grid_size, int tile_size,
    int padded_tile_size, int patch_size, idg::Coordinate patch_coordinate,
    cu::DeviceMemory& d_tile_ids, cu::DeviceMemory& d_tile_coordinates,
    cu::DeviceMemory& d_tiles, cu::DeviceMemory& d_patch) {
  compile_wtiling();
  kernel_wtiling_wtiles_from_patch_->enqueue(
      nr_polarizations, nr_tiles, grid_size, tile_size, padded_tile_size,
      patch_size, patch_coordinate, d_tile_ids, d_tile_coordinates, d_tiles,
      d_patch);
}

typedef struct {
  int nr_polarizations;
  int nr_timesteps;
  int nr_subgrids;
  std::shared_ptr<Report> report;
} ReportData;

void report_job(CUstream, CUresult, void* userData) {
  ReportData* data = static_cast<ReportData*>(userData);
  int nr_polarizations = data->nr_polarizations;
  int nr_correlations = nr_polarizations == 4 ? 4 : 2;
  int nr_timesteps = data->nr_timesteps;
  int nr_subgrids = data->nr_subgrids;
  data->report->print(nr_correlations, nr_timesteps, nr_subgrids);
  delete data;
}

ReportData* get_report_data(int nr_polarizations, int nr_timesteps,
                            int nr_subgrids, std::shared_ptr<Report> report) {
  ReportData* data = new ReportData();
  data->nr_polarizations = nr_polarizations;
  data->nr_timesteps = nr_timesteps;
  data->nr_subgrids = nr_subgrids;
  data->report = report;
  return data;
}

void InstanceCUDA::enqueue_report(cu::Stream& stream, int nr_polarizations,
                                  int nr_timesteps, int nr_subgrids) {
  ReportData* data =
      get_report_data(nr_polarizations, nr_timesteps, nr_subgrids, report_);
  stream.addCallback((CUstreamCallback)&report_job, data);
}

/*
 * Compilation
 */
void InstanceCUDA::compile_gridder(bool enable_avg_aterm) {
  if (!factory_gridder_) {
    factory_gridder_ = std::make_unique<KernelFactory<KernelGridder>>(
        *device_, KernelGridder::Parameters(enable_avg_aterm));
    kernel_gridder_ = factory_gridder_->create(*context_, *stream_execute_);
  }
}

void InstanceCUDA::compile_degridder() {
  if (!factory_degridder_) {
    factory_degridder_ =
        std::make_unique<KernelFactory<KernelDegridder>>(*device_);
    kernel_degridder_ = factory_degridder_->create(*context_, *stream_execute_);
  }
}

void InstanceCUDA::compile_scaler() {
  if (!factory_scaler_) {
    factory_scaler_ = std::make_unique<KernelFactory<KernelScaler>>(*device_);
    kernel_scaler_ = factory_scaler_->create(*context_, *stream_execute_);
  }
}

void InstanceCUDA::compile_adder() {
  if (!factory_adder_) {
    factory_adder_ = std::make_unique<KernelFactory<KernelAdder>>(*device_);
    kernel_adder_ = factory_adder_->create(*context_, *stream_execute_);
  }
}

void InstanceCUDA::compile_splitter() {
  if (!factory_splitter_) {
    factory_splitter_ =
        std::make_unique<KernelFactory<KernelSplitter>>(*device_);
    kernel_splitter_ = factory_splitter_->create(*context_, *stream_execute_);
  }
}

void InstanceCUDA::compile_calibrate() {
  if (!factory_calibrate_lmnp_) {
    factory_calibrate_lmnp_ =
        std::make_unique<KernelFactory<KernelCalibrateLMNP>>(*device_);
    kernel_calibrate_lmnp_ =
        factory_calibrate_lmnp_->create(*context_, *stream_execute_);
    factory_calibrate_sums_ =
        std::make_unique<KernelFactory<KernelCalibrateSums>>(*device_);
    kernel_calibrate_sums_ =
        factory_calibrate_sums_->create(*context_, *stream_execute_);
    factory_calibrate_gradient_ =
        std::make_unique<KernelFactory<KernelCalibrateGradient>>(*device_);
    kernel_calibrate_gradient_ =
        factory_calibrate_gradient_->create(*context_, *stream_execute_);
    factory_calibrate_hessian_ =
        std::make_unique<KernelFactory<KernelCalibrateHessian>>(*device_);
    kernel_calibrate_hessian_ =
        factory_calibrate_hessian_->create(*context_, *stream_execute_);
  }
}

void InstanceCUDA::compile_fft_shift() {
  if (!factory_fft_shift_) {
    factory_fft_shift_ =
        std::make_unique<KernelFactory<KernelFFTShift>>(*device_);
    kernel_fft_shift_ = factory_fft_shift_->create(*context_, *stream_execute_);
  }
}

void InstanceCUDA::compile_average_beam() {
  if (!factory_average_beam_) {
    factory_average_beam_ =
        std::make_unique<KernelFactory<KernelAverageBeam>>(*device_);
    kernel_average_beam_ =
        factory_average_beam_->create(*context_, *stream_execute_);
  }
}

void InstanceCUDA::compile_wtiling() {
  if (!factory_wtiling_copy_) {
    factory_wtiling_copy_ =
        std::make_unique<KernelFactory<KernelWTilingCopy>>(*device_);
    kernel_wtiling_copy_ =
        factory_wtiling_copy_->create(*context_, *stream_execute_);
    factory_wtiling_phasor_ =
        std::make_unique<KernelFactory<KernelWTilingPhasor>>(*device_);
    kernel_wtiling_phasor_ =
        factory_wtiling_phasor_->create(*context_, *stream_execute_);
    factory_wtiling_subgrids_from_wtiles_ =
        std::make_unique<KernelFactory<KernelWTilingSubgridsFromWtiles>>(
            *device_);
    kernel_wtiling_subgrids_from_wtiles_ =
        factory_wtiling_subgrids_from_wtiles_->create(*context_,
                                                      *stream_execute_);
    factory_wtiling_subgrids_to_wtiles_ =
        std::make_unique<KernelFactory<KernelWTilingSubgridsToWtiles>>(
            *device_);
    kernel_wtiling_subgrids_to_wtiles_ =
        factory_wtiling_subgrids_to_wtiles_->create(*context_,
                                                    *stream_execute_);
    factory_wtiling_wtiles_from_patch_ =
        std::make_unique<KernelFactory<KernelWTilingWTilesFromPatch>>(*device_);
    kernel_wtiling_wtiles_from_patch_ =
        factory_wtiling_wtiles_from_patch_->create(*context_, *stream_execute_);
    factory_wtiling_wtiles_to_patch_ =
        std::make_unique<KernelFactory<KernelWTilingWtilesToPatch>>(*device_);
    kernel_wtiling_wtiles_to_patch_ =
        factory_wtiling_wtiles_to_patch_->create(*context_, *stream_execute_);
    factory_wtiling_wtiles_from_grid_ =
        std::make_unique<KernelFactory<KernelWTilingWtilesFromGrid>>(*device_);
    kernel_wtiling_wtiles_from_grid_ =
        factory_wtiling_wtiles_from_grid_->create(*context_, *stream_execute_);
    factory_wtiling_wtiles_to_grid_ =
        std::make_unique<KernelFactory<KernelWTilingWtilesToGrid>>(*device_);
    kernel_wtiling_wtiles_to_grid_ =
        factory_wtiling_wtiles_to_grid_->create(*context_, *stream_execute_);
  }
}

/*
 * Event destructor
 */
void InstanceCUDA::free_events() { events.clear(); }

/*
 * Device interface
 */
size_t InstanceCUDA::get_free_memory() const {
  return context_->getFreeMemory();
}

size_t InstanceCUDA::get_total_memory() const {
  return context_->getTotalMemory();
}

}  // end namespace idg::kernel::cuda
