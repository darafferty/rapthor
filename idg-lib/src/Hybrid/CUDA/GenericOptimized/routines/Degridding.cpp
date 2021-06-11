#include <mutex>

#include "../GenericOptimized.h"
#include "InstanceCUDA.h"

using namespace idg::proxy::cuda;
using namespace idg::proxy::cpu;
using namespace idg::kernel::cpu;
using namespace idg::kernel::cuda;
using namespace powersensor;

namespace idg {
namespace proxy {
namespace hybrid {

void GenericOptimized::run_degridding(
    const Plan& plan, const Array1D<float>& frequencies,
    Array3D<Visibility<std::complex<float>>>& visibilities,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Grid& grid, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterms_offsets,
    const Array2D<float>& spheroidal) {
#if defined(DEBUG)
  std::cout << "GenericOptimized::" << __func__ << std::endl;
#endif

  InstanceCUDA& device = get_device(0);
  const cu::Context& context = device.get_context();

  InstanceCPU& cpuKernels = cpuProxy->get_kernels();

  // Arguments
  auto nr_baselines = visibilities.get_z_dim();
  auto nr_timesteps = visibilities.get_y_dim();
  auto nr_channels = visibilities.get_x_dim();
  auto nr_stations = aterms.get_z_dim();
  auto grid_size = grid.get_x_dim();
  auto cell_size = plan.get_cell_size();
  auto image_size = cell_size * grid_size;
  auto subgrid_size = plan.get_subgrid_size();
  auto w_step = plan.get_w_step();
  auto& shift = plan.get_shift();

  WTileUpdateSet wtile_initialize_set = plan.get_wtile_initialize_set();

  // Configuration
  const unsigned nr_devices = get_num_devices();
  int device_id = 0;  // only one GPU is used
  int jobsize = m_gridding_state.jobsize[0];

  // Page-locked host memory
  cu::RegisteredMemory h_metadata(context, (void*)plan.get_metadata_ptr(),
                                  plan.get_sizeof_metadata());
  auto max_nr_subgrids = plan.get_max_nr_subgrids(jobsize);
  auto sizeof_subgrids =
      auxiliary::sizeof_subgrids(max_nr_subgrids, subgrid_size);
  cu::HostMemory& h_subgrids = *m_buffers.h_subgrids;
  h_subgrids.resize(sizeof_subgrids);

  // Performance measurements
  m_report->initialize(nr_channels, subgrid_size, grid_size);
  device.set_report(m_report);
  cpuKernels.set_report(m_report);
  std::vector<State> startStates(nr_devices + 1);
  std::vector<State> endStates(nr_devices + 1);

  // Events
  std::vector<std::unique_ptr<cu::Event>> inputCopied;
  std::vector<std::unique_ptr<cu::Event>> gpuFinished;
  std::vector<std::unique_ptr<cu::Event>> outputCopied;
  for (unsigned bl = 0; bl < nr_baselines; bl += jobsize) {
    inputCopied.push_back(std::unique_ptr<cu::Event>(new cu::Event(context)));
    gpuFinished.push_back(std::unique_ptr<cu::Event>(new cu::Event(context)));
    outputCopied.push_back(std::unique_ptr<cu::Event>(new cu::Event(context)));
  }

  // Load memory objects
  cu::DeviceMemory& d_wavenumbers = *m_buffers.d_wavenumbers;
  cu::DeviceMemory& d_spheroidal = *m_buffers.d_spheroidal;
  cu::DeviceMemory& d_aterms = *m_buffers.d_aterms;
  cu::DeviceMemory& d_aterms_indices = *m_buffers.d_aterms_indices_[0];

  // Load streams
  cu::Stream& executestream = device.get_execute_stream();
  cu::Stream& htodstream = device.get_htod_stream();
  cu::Stream& dtohstream = device.get_dtoh_stream();

  // Start performance measurement
  startStates[device_id] = device.measure();
  startStates[nr_devices] = hostPowerSensor->read();

  // Locks to make the GPU wait
  std::vector<std::mutex> locks_gpu(jobs.size());
  for (auto& lock : locks_gpu) {
    lock.lock();
  }

  // Locks to make the CPU wait
  std::vector<std::mutex> locks_cpu(jobs.size());
  for (auto& lock : locks_cpu) {
    lock.lock();
  }

  // Start host thread to create subgrids
  std::thread host_thread = std::thread([&] {
    int subgrid_offset = 0;
    for (unsigned job_id = 0; job_id < jobs.size(); job_id++) {
      // Get parameters for current job
      auto current_nr_subgrids = jobs[job_id].current_nr_subgrids;
      void* metadata_ptr = jobs[job_id].metadata_ptr;
      std::complex<float>* grid_ptr = grid.data();
      unsigned local_id = job_id % 2;

      // Load memory objects
      cu::DeviceMemory& d_subgrids = *m_buffers.d_subgrids_[local_id];
      cu::DeviceMemory& d_metadata = *m_buffers.d_metadata_[local_id];

      // Wait for input buffer to be free
      if (job_id > 0) {
        locks_cpu[job_id - 1].lock();
      }

      // Run splitter kernel
      cu::Marker marker_splitter("run_splitter", cu::Marker::blue);
      marker_splitter.start();

      if (plan.get_use_wtiles()) {
        if (!m_disable_wtiling_gpu)
        {
          run_subgrids_from_wtiles(subgrid_offset, current_nr_subgrids, subgrid_size,
                                   image_size, w_step, shift, wtile_initialize_set,
                                   d_subgrids, d_metadata);
        } else {
          cpuKernels.run_splitter_wtiles(
              current_nr_subgrids, grid_size, subgrid_size, image_size, w_step,
              shift.data(), subgrid_offset, wtile_initialize_set, metadata_ptr,
              h_subgrids, grid_ptr);
        }
        subgrid_offset += current_nr_subgrids;
      } else if (w_step != 0.0) {
        cpuKernels.run_splitter_wstack(current_nr_subgrids, grid_size,
                                       subgrid_size, metadata_ptr, h_subgrids,
                                       grid_ptr);
      } else {
        cpuKernels.run_splitter(current_nr_subgrids, grid_size, subgrid_size,
                                metadata_ptr, h_subgrids, grid_ptr);
      }

      marker_splitter.end();

      // Copy subgrids to device
      auto sizeof_subgrids =
          auxiliary::sizeof_subgrids(current_nr_subgrids, subgrid_size);
      htodstream.memcpyHtoDAsync(d_subgrids, h_subgrids, sizeof_subgrids);

      // Wait for subgrids to be copied
      htodstream.synchronize();

      // Unlock this job
      locks_gpu[job_id].unlock();
    }
  });  // end host thread

  // Iterate all jobs
  for (unsigned job_id = 0; job_id < jobs.size(); job_id++) {
    // Id for double-buffering
    unsigned local_id = job_id % 2;
    unsigned job_id_next = job_id + 1;
    unsigned local_id_next = (local_id + 1) % 2;

    // Get parameters for current job
    auto current_time_offset = jobs[job_id].current_time_offset;
    auto current_nr_baselines = jobs[job_id].current_nr_baselines;
    auto current_nr_subgrids = jobs[job_id].current_nr_subgrids;
    void* metadata_ptr = jobs[job_id].metadata_ptr;
    void* uvw_ptr = jobs[job_id].uvw_ptr;
    void* visibilities_ptr = jobs[job_id].visibilities_ptr;

    // Load memory objects
    cu::DeviceMemory& d_visibilities = *m_buffers.d_visibilities_[local_id];
    cu::DeviceMemory& d_uvw = *m_buffers.d_uvw_[local_id];
    cu::DeviceMemory& d_subgrids = *m_buffers.d_subgrids_[local_id];
    cu::DeviceMemory& d_metadata = *m_buffers.d_metadata_[local_id];

    // Wait for subgrids to be computed
    locks_gpu[job_id].lock();

    // Copy input data for first job to device
    if (job_id == 0) {
      auto sizeof_uvw =
          auxiliary::sizeof_uvw(current_nr_baselines, nr_timesteps);
      auto sizeof_metadata = auxiliary::sizeof_metadata(current_nr_subgrids);
      htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, sizeof_uvw);
      htodstream.memcpyHtoDAsync(d_metadata, metadata_ptr, sizeof_metadata);
      htodstream.record(*inputCopied[job_id]);
    }

    // Copy input data for next job
    if (job_id_next < jobs.size()) {
      // Load memory objects
      cu::DeviceMemory& d_uvw_next = *m_buffers.d_uvw_[local_id_next];
      cu::DeviceMemory& d_metadata_next = *m_buffers.d_metadata_[local_id_next];

      // Get parameters for next job
      auto nr_baselines_next = jobs[job_id_next].current_nr_baselines;
      auto nr_subgrids_next = jobs[job_id_next].current_nr_subgrids;
      void* metadata_ptr_next = jobs[job_id_next].metadata_ptr;
      void* uvw_ptr_next = jobs[job_id_next].uvw_ptr;

      // Copy input data to device
      auto sizeof_uvw_next =
          auxiliary::sizeof_uvw(nr_baselines_next, nr_timesteps);
      auto sizeof_metadata_next = auxiliary::sizeof_metadata(nr_subgrids_next);
      htodstream.memcpyHtoDAsync(d_uvw_next, uvw_ptr_next, sizeof_uvw_next);
      htodstream.memcpyHtoDAsync(d_metadata_next, metadata_ptr_next,
                                 sizeof_metadata_next);
      htodstream.record(*inputCopied[job_id_next]);
    }

    // Wait for input to be copied
    executestream.waitEvent(*inputCopied[job_id]);

    // Wait for output buffer to be free
    if (job_id > 1) {
      executestream.waitEvent(*outputCopied[job_id - 2]);
    }

    // Initialize visibilities to zero
    d_visibilities.zero(executestream);

    // Launch FFT
    device.launch_subgrid_fft(d_subgrids, current_nr_subgrids,
                              ImageDomainToFourierDomain);

    // Launch degridder kernel
    device.launch_degridder(current_time_offset, current_nr_subgrids, grid_size,
                            subgrid_size, image_size, w_step, nr_channels,
                            nr_stations, shift(0), shift(1), d_uvw,
                            d_wavenumbers, d_visibilities, d_spheroidal,
                            d_aterms, d_aterms_indices, d_metadata, d_subgrids);
    executestream.record(*gpuFinished[job_id]);

    // Signal that the input buffer is free
    inputCopied[job_id]->synchronize();
    locks_cpu[job_id].unlock();

    // Wait for degridder to finish
    gpuFinished[job_id]->synchronize();

    // Copy visibilities to host
    dtohstream.waitEvent(*gpuFinished[job_id]);
    auto sizeof_visibilities = auxiliary::sizeof_visibilities(
        current_nr_baselines, nr_timesteps, nr_channels);
    dtohstream.memcpyDtoHAsync(visibilities_ptr, d_visibilities,
                               sizeof_visibilities);
    dtohstream.record(*outputCopied[job_id]);

    // Report performance
    device.enqueue_report(executestream, jobs[job_id].current_nr_timesteps,
                          jobs[job_id].current_nr_subgrids);
  }  // end for bl

  // Wait for host thread
  if (host_thread.joinable()) {
    host_thread.join();
  }

  // Wait for all visibilities to be copied
  dtohstream.synchronize();

  // End performance measurement
  endStates[device_id] = device.measure();
  endStates[nr_devices] = hostPowerSensor->read();
  m_report->update(Report::host, startStates[nr_devices],
                   endStates[nr_devices]);

  // Update report
  auto total_nr_subgrids = plan.get_nr_subgrids();
  auto total_nr_timesteps = plan.get_nr_timesteps();
  auto total_nr_visibilities = plan.get_nr_visibilities();
  m_report->print_total(total_nr_timesteps, total_nr_subgrids);
  m_report->print_visibilities(auxiliary::name_degridding,
                               total_nr_visibilities);
}  // end run_degridding

void GenericOptimized::do_degridding(
    const Plan& plan, const Array1D<float>& frequencies,
    Array3D<Visibility<std::complex<float>>>& visibilities,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterms_offsets,
    const Array2D<float>& spheroidal) {
#if defined(DEBUG)
  std::cout << "GenericOptimized::" << __func__ << std::endl;
  std::clog << "### Initialize degridding" << std::endl;
#endif
  CUDA::initialize(plan, frequencies, visibilities, uvw, baselines, aterms,
                   aterms_offsets, spheroidal);

#if defined(DEBUG)
  std::clog << "### Run degridding" << std::endl;
#endif
  run_degridding(plan, frequencies, visibilities, uvw, baselines, *m_grid,
                 aterms, aterms_offsets, spheroidal);

#if defined(DEBUG)
  std::clog << "### Finish degridding" << std::endl;
#endif
}  // end do_degridding

}  // namespace hybrid
}  // namespace proxy
}  // namespace idg
