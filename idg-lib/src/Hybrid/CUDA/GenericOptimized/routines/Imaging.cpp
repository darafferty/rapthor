#include <cudawrappers/cu.hpp>
#include <cudawrappers/nvtx.hpp>

#include "../GenericOptimized.h"
#include "InstanceCUDA.h"

using namespace idg::proxy::cuda;
using namespace idg::proxy::cpu;
using namespace idg::kernel::cpu;
using namespace idg::kernel::cuda;

namespace idg {
namespace proxy {
namespace hybrid {

void GenericOptimized::run_imaging(
    const Plan& plan, const aocommon::xt::Span<float, 1>& frequencies,
    aocommon::xt::Span<std::complex<float>, 4>& visibilities,
    const aocommon::xt::Span<UVW<float>, 2>& uvw,
    const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
        baselines,
    aocommon::xt::Span<std::complex<float>, 4>& grid,
    const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
    const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
    const aocommon::xt::Span<float, 2>& taper, ImagingMode mode) {
  InstanceCUDA& device = get_device();

  auto cpuKernels = cpuProxy->get_kernels();

  // Arguments
  const size_t nr_baselines = visibilities.shape(0);
  const size_t nr_timesteps = visibilities.shape(1);
  const size_t nr_channels = visibilities.shape(2);
  const size_t nr_correlations = visibilities.shape(3);
  const size_t nr_stations = aterms.shape(1);
  const size_t nr_polarizations = grid.shape(1);
  const size_t grid_size = grid.shape(2);
  assert(grid.shape(3) == grid_size);
  const float cell_size = plan.get_cell_size();
  const float image_size = cell_size * grid_size;
  const size_t subgrid_size = plan.get_subgrid_size();
  const float w_step = plan.get_w_step();
  const std::array<float, 2>& shift = plan.get_shift();

  WTileUpdateSet wtile_set;
  if (mode == ImagingMode::mode_gridding) {
    wtile_set = plan.get_wtile_flush_set();
  } else if (mode == ImagingMode::mode_degridding) {
    wtile_set = plan.get_wtile_initialize_set();
  }

  Tensor<float, 1> wavenumbers = compute_wavenumbers(frequencies);

  const size_t sizeof_wavenumbers =
      wavenumbers.Span().size() * sizeof(*wavenumbers.Span().data());
  const size_t sizeof_taper = taper.size() * sizeof(*taper.data());
  const size_t sizeof_aterms = aterms.size() * sizeof(*aterms.data());
  const size_t sizeof_aterm_indices =
      auxiliary::sizeof_aterm_indices(nr_baselines, nr_timesteps);
  const unsigned int* aterm_indices_ptr = plan.get_aterm_indices_ptr();

  // Configuration
  const unsigned nr_devices = 1;
  const int device_id = 0;

  // Performance measurements
  get_report()->initialize(nr_channels, subgrid_size, grid_size);
  device.set_report(get_report());
  cpuKernels->set_report(get_report());
  std::vector<pmt::State> startStates(nr_devices + 1);
  std::vector<pmt::State> endStates(nr_devices + 1);

  // Load streams
  cu::Stream& executestream = device.get_execute_stream();
  cu::Stream& htodstream = device.get_htod_stream();
  cu::Stream& dtohstream = device.get_dtoh_stream();

  // Allocate device memory
  cu::DeviceMemory d_wavenumbers(sizeof_wavenumbers, CU_MEMORYTYPE_DEVICE);
  cu::DeviceMemory d_taper(sizeof_taper, CU_MEMORYTYPE_DEVICE);
  cu::DeviceMemory d_aterms(sizeof_aterms, CU_MEMORYTYPE_DEVICE);
  cu::DeviceMemory d_aterm_indices(sizeof_aterm_indices, CU_MEMORYTYPE_DEVICE);

  // Initialize device memory
  htodstream.memcpyHtoDAsync(d_wavenumbers, wavenumbers.Span().data(),
                             sizeof_wavenumbers);
  htodstream.memcpyHtoDAsync(d_taper, taper.data(), sizeof_taper);
  htodstream.memcpyHtoDAsync(d_aterms, aterms.data(), sizeof_aterms);
  htodstream.memcpyHtoDAsync(d_aterm_indices, aterm_indices_ptr,
                             sizeof_aterm_indices);

  // When degridding, d_avg_aterm is not used and remains a null pointer.
  // When gridding, d_avg_aterm always holds a cu::DeviceMemory object. When
  // average aterm correction is disabled, the cu::DeviceMemory object contains
  // a null pointer, such that the gridder kernel can detect that it should not
  // apply average aterm corrections.
  std::unique_ptr<cu::DeviceMemory> d_avg_aterm;
  if (mode == ImagingMode::mode_gridding) {
    size_t sizeof_avg_aterm_correction =
        m_avg_aterm_correction.size() * sizeof(std::complex<float>);
    d_avg_aterm.reset(new cu::DeviceMemory(sizeof_avg_aterm_correction,
                                           CU_MEMORYTYPE_DEVICE));
    htodstream.memcpyHtoDAsync(*d_avg_aterm, m_avg_aterm_correction.data(),
                               sizeof_avg_aterm_correction);
  }

  // Get the available device memory, reserving some
  // space for the w-padded tile FFT plan which is allocated
  // in run_wtiles_to_grid and run_wtiles_from_grid.
  size_t bytes_free = device.get_free_memory();
  if (!m_disable_wtiling && !m_disable_wtiling_gpu) {
    bytes_free -= bytes_required_wtiling(plan.get_wtile_initialize_set(),
                                         nr_polarizations, subgrid_size,
                                         image_size, w_step, shift, bytes_free);
  }

  // Initialize jobs
  std::vector<JobData> jobs;
  int jobsize =
      initialize_jobs(nr_baselines, nr_timesteps, nr_channels, subgrid_size,
                      bytes_free, plan, visibilities, uvw, jobs);

  // Allocate device memory for jobs
  const int max_nr_subgrids = plan.get_max_nr_subgrids(jobsize);
  size_t sizeof_visibilities = auxiliary::sizeof_visibilities(
      jobsize, nr_timesteps, nr_channels, nr_correlations);
  size_t sizeof_uvw = auxiliary::sizeof_uvw(jobsize, nr_timesteps);
  size_t sizeof_subgrids = auxiliary::sizeof_subgrids(
      max_nr_subgrids, subgrid_size, nr_correlations);
  size_t sizeof_metadata = auxiliary::sizeof_metadata(max_nr_subgrids);

  std::array<cu::DeviceMemory, 2> d_visibilities_{
      cu::DeviceMemory(sizeof_visibilities, CU_MEMORYTYPE_DEVICE),
      cu::DeviceMemory(sizeof_visibilities, CU_MEMORYTYPE_DEVICE)};
  std::array<cu::DeviceMemory, 2> d_uvw_{
      cu::DeviceMemory(sizeof_uvw, CU_MEMORYTYPE_DEVICE),
      cu::DeviceMemory(sizeof_uvw, CU_MEMORYTYPE_DEVICE)};
  std::array<cu::DeviceMemory, 2> d_subgrids_{
      cu::DeviceMemory(sizeof_subgrids, CU_MEMORYTYPE_DEVICE),
      cu::DeviceMemory(sizeof_subgrids, CU_MEMORYTYPE_DEVICE)};
  std::array<cu::DeviceMemory, 2> d_metadata_{
      cu::DeviceMemory(sizeof_metadata, CU_MEMORYTYPE_DEVICE),
      cu::DeviceMemory(sizeof_metadata, CU_MEMORYTYPE_DEVICE)};

  // Plan subgrid fft
  std::unique_ptr<KernelFFT> fft_kernel =
      device.plan_batched_fft(subgrid_size, max_nr_subgrids * nr_polarizations);

  // Page-locked host memory
  cu::HostMemory h_metadata(
      const_cast<void*>(reinterpret_cast<const void*>(plan.get_metadata_ptr())),
      plan.get_sizeof_metadata());
  cu::HostMemory h_subgrids(!m_disable_wtiling && !m_disable_wtiling_gpu
                                ? 0
                                : auxiliary::sizeof_subgrids(max_nr_subgrids,
                                                             subgrid_size,
                                                             nr_correlations));

  // Events
  const unsigned int nr_jobs = (nr_baselines + jobsize - 1) / jobsize;
  std::vector<cu::Event> inputCopied(nr_jobs);
  std::vector<cu::Event> gpuFinished(nr_jobs);
  std::vector<cu::Event> outputCopied(nr_jobs);  // Only used when degridding.

  // Start performance measurement
  startStates[device_id] = device.measure();
  startStates[nr_devices] = power_meter_->Read();

  // Iterate all jobs
  for (unsigned job_id = 0; job_id < jobs.size(); job_id++) {
    // Id for double-buffering
    const unsigned local_id = job_id % 2;
    const unsigned job_id_next = job_id + 1;
    const unsigned local_id_next = (local_id + 1) % 2;

    // Get parameters for current job
    unsigned int time_offset_current = jobs[job_id].current_time_offset;
    unsigned int nr_baselines_current = jobs[job_id].current_nr_baselines;
    unsigned int nr_subgrids_current = jobs[job_id].current_nr_subgrids;
    const Metadata* metadata_ptr = jobs[job_id].metadata_ptr;
    const UVW<float>* uvw_ptr = jobs[job_id].uvw_ptr;
    std::complex<float>* visibilities_ptr = jobs[job_id].visibilities_ptr;
    std::complex<float>* subgrids_ptr = h_subgrids;

    // Load memory objects
    cu::DeviceMemory& d_visibilities = d_visibilities_[local_id];
    cu::DeviceMemory& d_uvw = d_uvw_[local_id];
    cu::DeviceMemory& d_subgrids = d_subgrids_[local_id];
    cu::DeviceMemory& d_metadata = d_metadata_[local_id];

    // Copy input data for first job to device
    if (job_id == 0) {
      if (mode == ImagingMode::mode_gridding) {
        htodstream.memcpyHtoDAsync(
            d_visibilities, visibilities_ptr,
            auxiliary::sizeof_visibilities(nr_baselines_current, nr_timesteps,
                                           nr_channels, nr_correlations));
      }
      htodstream.memcpyHtoDAsync(
          d_uvw, uvw_ptr,
          auxiliary::sizeof_uvw(nr_baselines_current, nr_timesteps));
      htodstream.memcpyHtoDAsync(
          d_metadata, metadata_ptr,
          auxiliary::sizeof_metadata(nr_subgrids_current));
      htodstream.record(inputCopied[job_id]);
    }

    // Copy input data for next job
    if (job_id_next < jobs.size()) {
      // Wait for previous job to finish before
      // overwriting its input buffers.
      if (job_id_next > 1) {
        htodstream.wait(gpuFinished[job_id_next - 2]);
      }

      // Load memory objects
      cu::DeviceMemory& d_visibilities_next = d_visibilities_[local_id_next];
      cu::DeviceMemory& d_uvw_next = d_uvw_[local_id_next];
      cu::DeviceMemory& d_metadata_next = d_metadata_[local_id_next];

      // Get parameters for next job
      unsigned int nr_baselines_next = jobs[job_id_next].current_nr_baselines;
      unsigned int nr_subgrids_next = jobs[job_id_next].current_nr_subgrids;
      const Metadata* metadata_ptr_next = jobs[job_id_next].metadata_ptr;
      const UVW<float>* uvw_ptr_next = jobs[job_id_next].uvw_ptr;
      std::complex<float>* visibilities_ptr_next =
          jobs[job_id_next].visibilities_ptr;

      // Copy input data to device
      if (mode == ImagingMode::mode_gridding) {
        size_t sizeof_visibilities_next = auxiliary::sizeof_visibilities(
            nr_baselines_next, nr_timesteps, nr_channels, nr_correlations);
        htodstream.memcpyHtoDAsync(d_visibilities_next, visibilities_ptr_next,
                                   sizeof_visibilities_next);
      }
      htodstream.memcpyHtoDAsync(
          d_uvw_next, uvw_ptr_next,
          auxiliary::sizeof_uvw(nr_baselines_next, nr_timesteps));
      htodstream.memcpyHtoDAsync(d_metadata_next, metadata_ptr_next,
                                 auxiliary::sizeof_metadata(nr_subgrids_next));
      htodstream.record(inputCopied[job_id_next]);
    }

    // Initialize output buffer to zero
    const size_t sizeof_subgrids = auxiliary::sizeof_subgrids(
        nr_subgrids_current, subgrid_size, nr_correlations);
    if (mode == ImagingMode::mode_gridding) {
      executestream.zero(d_subgrids, sizeof_subgrids);
    } else if (mode == ImagingMode::mode_degridding) {
      if (job_id > 1) {
        executestream.wait(outputCopied[job_id - 2]);
      }
      executestream.zero(d_visibilities, sizeof_visibilities);
    }

    // Wait for input to be copied
    executestream.wait(inputCopied[job_id]);

    if (mode == ImagingMode::mode_gridding) {
      // Launch gridder kernel
      device.launch_gridder(
          time_offset_current, nr_subgrids_current, nr_polarizations, grid_size,
          subgrid_size, image_size, w_step, nr_channels, nr_stations, shift[0],
          shift[1], d_uvw, d_wavenumbers, d_visibilities, d_taper, d_aterms,
          d_aterm_indices, d_metadata, *d_avg_aterm, d_subgrids);

      // Launch FFT
      device.launch_batched_fft(*fft_kernel, d_subgrids,
                                nr_subgrids_current * nr_polarizations,
                                FourierDomainToImageDomain);

      // In case of W-Tiling, adder_subgrids_to_wtiles performs scaling of the
      // subgrids, otherwise the scaler kernel needs to be called here.
      if (!plan.get_use_wtiles()) {
        device.launch_scaler(nr_subgrids_current, nr_polarizations,
                             subgrid_size, d_subgrids);
      }
      executestream.record(gpuFinished[job_id]);

      // Copy subgrid to host
      if (m_disable_wtiling || m_disable_wtiling_gpu) {
        dtohstream.wait(gpuFinished[job_id]);
        dtohstream.memcpyDtoHAsync(h_subgrids, d_subgrids, sizeof_subgrids);
        dtohstream.synchronize();
      }

      // Run adder kernel
      nvtx::Marker marker_adder("run_adder", nvtx::Marker::blue);
      marker_adder.start();

      if (plan.get_use_wtiles()) {
        auto subgrid_offset = plan.get_subgrid_offset(jobs[job_id].first_bl);

        if (!m_disable_wtiling_gpu) {
          run_subgrids_to_wtiles(nr_polarizations, subgrid_offset,
                                 nr_subgrids_current, subgrid_size, image_size,
                                 w_step, shift, wtile_set, d_subgrids,
                                 d_metadata);
        } else {
          cpuKernels->run_adder_wtiles(
              nr_subgrids_current, nr_polarizations, grid_size, subgrid_size,
              image_size, w_step, shift.data(), subgrid_offset, wtile_set,
              metadata_ptr, subgrids_ptr, get_grid().data());
        }
      } else if (w_step != 0.0) {
        cpuKernels->run_adder_wstack(nr_subgrids_current, nr_polarizations,
                                     grid_size, subgrid_size, metadata_ptr,
                                     subgrids_ptr, get_grid().data());
      } else {
        cpuKernels->run_adder(nr_subgrids_current, nr_polarizations, grid_size,
                              subgrid_size, metadata_ptr, subgrids_ptr,
                              get_grid().data());
      }

      marker_adder.end();
    } else if (mode == ImagingMode::mode_degridding) {
      // Run splitter kernel
      nvtx::Marker marker_splitter("run_splitter", nvtx::Marker::blue);
      marker_splitter.start();

      if (plan.get_use_wtiles()) {
        auto subgrid_offset = plan.get_subgrid_offset(jobs[job_id].first_bl);

        if (!m_disable_wtiling_gpu) {
          run_subgrids_from_wtiles(nr_polarizations, subgrid_offset,
                                   nr_subgrids_current, subgrid_size,
                                   image_size, w_step, shift, wtile_set,
                                   d_subgrids, d_metadata);
        } else {
          cpuKernels->run_splitter_wtiles(
              nr_subgrids_current, nr_polarizations, grid_size, subgrid_size,
              image_size, w_step, shift.data(), subgrid_offset, wtile_set,
              metadata_ptr, subgrids_ptr, get_grid().data());
        }
      } else if (w_step != 0.0) {
        cpuKernels->run_splitter_wstack(nr_subgrids_current, nr_polarizations,
                                        grid_size, subgrid_size, metadata_ptr,
                                        subgrids_ptr, get_grid().data());
      } else {
        cpuKernels->run_splitter(nr_subgrids_current, nr_polarizations,
                                 grid_size, subgrid_size, metadata_ptr,
                                 subgrids_ptr, get_grid().data());
      }

      if (m_disable_wtiling || m_disable_wtiling_gpu) {
        // Copy subgrids to device
        htodstream.memcpyHtoDAsync(d_subgrids, subgrids_ptr, sizeof_subgrids);
      }

      marker_splitter.end();

      // Launch FFT
      device.launch_batched_fft(*fft_kernel, d_subgrids,
                                nr_subgrids_current * nr_polarizations,
                                ImageDomainToFourierDomain);

      // Launch degridder kernel
      device.launch_degridder(
          time_offset_current, nr_subgrids_current, nr_polarizations, grid_size,
          subgrid_size, image_size, w_step, nr_channels, nr_stations, shift[0],
          shift[1], d_uvw, d_wavenumbers, d_visibilities, d_taper, d_aterms,
          d_aterm_indices, d_metadata, d_subgrids);
      executestream.record(gpuFinished[job_id]);

      // Copy visibilities to host
      dtohstream.wait(gpuFinished[job_id]);
      dtohstream.memcpyDtoHAsync(
          visibilities_ptr, d_visibilities,
          auxiliary::sizeof_visibilities(nr_baselines_current, nr_timesteps,
                                         nr_channels, nr_correlations));
      dtohstream.record(outputCopied[job_id]);
    }

    // Report performance
    device.enqueue_report(executestream, nr_polarizations,
                          jobs[job_id].current_nr_timesteps,
                          jobs[job_id].current_nr_subgrids);
  }  // end for bl

  // Wait for all visibilities to be copied
  if (mode == ImagingMode::mode_degridding) {
    dtohstream.synchronize();
  }

  // Wait for all reports to be printed
  executestream.synchronize();

  // End performance measurement
  endStates[device_id] = device.measure();
  endStates[nr_devices] = power_meter_->Read();
  get_report()->update(Report::device, startStates[device_id],
                       endStates[device_id]);
  get_report()->update(Report::host, startStates[nr_devices],
                       endStates[nr_devices]);

  // Update report
  auto total_nr_subgrids = plan.get_nr_subgrids();
  auto total_nr_timesteps = plan.get_nr_timesteps();
  auto total_nr_visibilities = plan.get_nr_visibilities();
  get_report()->print_total(nr_correlations, total_nr_timesteps,
                            total_nr_subgrids);
  const std::string* name;
  if (mode == ImagingMode::mode_gridding) {
    name = &auxiliary::name_gridding;
  } else if (mode == ImagingMode::mode_degridding) {
    name = &auxiliary::name_degridding;
  }
  get_report()->print_visibilities(*name, total_nr_visibilities);
}

}  // end namespace hybrid
}  // end namespace proxy
}  // end namespace idg
