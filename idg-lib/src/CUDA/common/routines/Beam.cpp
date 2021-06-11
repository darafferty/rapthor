#include "../CUDA.h"
#include "../InstanceCUDA.h"

using namespace idg::kernel::cuda;

namespace idg {
namespace proxy {
namespace cuda {

void CUDA::do_compute_avg_beam(
    const unsigned int nr_antennas, const unsigned int nr_channels,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Array4D<Matrix2x2<std::complex<float>>>& aterms,
    const Array1D<unsigned int>& aterms_offsets, const Array4D<float>& weights,
    idg::Array4D<std::complex<float>>& average_beam) {
#if defined(DEBUG)
  std::cout << "CUDA::" << __func__ << std::endl;
#endif

  const unsigned int nr_aterms = aterms_offsets.size() - 1;
  const unsigned int nr_baselines = baselines.get_x_dim();
  const unsigned int nr_timesteps = uvw.get_x_dim();
  const unsigned int subgrid_size = average_beam.get_w_dim();

  InstanceCUDA& device = get_device(0);
  cu::Context& context = device.get_context();

  // Performance reporting
  m_report->initialize();
  device.set_report(m_report);

  // Allocate static device memory
  cu::DeviceMemory& d_aterms = *m_buffers.d_aterms;
  d_aterms.resize(aterms.bytes());
  cu::DeviceMemory d_baselines(context, baselines.bytes());
  cu::DeviceMemory d_aterms_offsets(context, aterms_offsets.bytes());
  cu::DeviceMemory d_average_beam(
      context, average_beam.bytes() * 2);  // double-precision!

  // Set jobsize and allocate dynamic memory (per thread)
  int jobsize = baselines.size() / 2;
  do {
    size_t bytes_free = device.get_free_memory();
    size_t sizeof_uvw = auxiliary::sizeof_uvw(jobsize, nr_timesteps);
    size_t sizeof_weights =
        auxiliary::sizeof_weights(jobsize, nr_timesteps, nr_channels);
    size_t bytes_required = 2 * sizeof_uvw + 2 * sizeof_weights;

    // Try to allocate memory
    if (bytes_free > bytes_required) {
      for (unsigned int i = 0; i < 2; i++) {
        m_buffers.d_uvw_[i]->resize(sizeof_uvw);
        m_buffers.d_visibilities_[i]->resize(
            sizeof_weights);  // visibilities buffer!
      }
      break;
    } else {
      jobsize /= 2;
    }
  } while (jobsize > 0);

  if (jobsize == 0) {
    throw std::runtime_error(
        "Could not allocate memory for average beam kernel.");
  }

  // Initialize job data
  struct JobData {
    unsigned int current_nr_baselines;
    void* uvw_ptr;
    void* weights_ptr;
  };
  std::vector<JobData> jobs;
  for (unsigned bl = 0; bl < nr_baselines; bl += jobsize) {
    JobData job;
    unsigned int first_bl = bl;
    unsigned int last_bl = std::min(bl + jobsize, nr_baselines);
    job.current_nr_baselines = last_bl - first_bl;
    job.uvw_ptr = uvw.data(first_bl, 0);
    job.weights_ptr = weights.data(first_bl, 0, 0, 0);
    if (job.current_nr_baselines > 0) {
      jobs.push_back(job);
    }
  }

  // Events
  std::vector<std::unique_ptr<cu::Event>> inputCopied;
  for (unsigned bl = 0; bl < nr_baselines; bl += jobsize) {
    inputCopied.push_back(std::unique_ptr<cu::Event>(
        new cu::Event(context, CU_EVENT_BLOCKING_SYNC)));
  }

  // Load streams
  cu::Stream& htodstream = device.get_htod_stream();
  cu::Stream& dtohstream = device.get_dtoh_stream();
  cu::Stream& executestream = device.get_execute_stream();

  // Copy static data
  htodstream.memcpyHtoDAsync(d_aterms, aterms.data(), aterms.bytes());
  htodstream.memcpyHtoDAsync(d_baselines, baselines.data(), baselines.bytes());
  htodstream.memcpyHtoDAsync(d_aterms_offsets, aterms_offsets.data(),
                             aterms_offsets.bytes());

  // Initialize average beam
  d_average_beam.zero(htodstream);

  for (unsigned int job_id = 0; job_id < jobs.size(); job_id++) {
    // Id for double-buffering
    unsigned local_id = job_id % 2;
    unsigned job_id_next = job_id + 1;
    unsigned local_id_next = (local_id + 1) % 2;

    auto& job = jobs[job_id];
    cu::DeviceMemory& d_uvw = *m_buffers.d_uvw_[local_id];
    cu::DeviceMemory& d_weights = *m_buffers.d_visibilities_[local_id];

    // Copy input for first job
    if (job_id == 0) {
      auto sizeof_uvw =
          auxiliary::sizeof_uvw(job.current_nr_baselines, nr_timesteps);
      auto sizeof_weights = auxiliary::sizeof_weights(
          job.current_nr_baselines, nr_timesteps, nr_channels);
      htodstream.memcpyHtoDAsync(d_uvw, job.uvw_ptr, sizeof_uvw);
      htodstream.memcpyHtoDAsync(d_weights, job.weights_ptr, sizeof_weights);
      htodstream.record(*inputCopied[job_id]);
    }

    // Copy input for next job (if any)
    if (job_id_next < jobs.size()) {
      auto& job_next = jobs[job_id_next];
      cu::DeviceMemory& d_uvw_next = *m_buffers.d_uvw_[local_id_next];
      cu::DeviceMemory& d_weights_next =
          *m_buffers.d_visibilities_[local_id_next];
      auto sizeof_uvw =
          auxiliary::sizeof_uvw(job_next.current_nr_baselines, nr_timesteps);
      auto sizeof_weights = auxiliary::sizeof_weights(
          job_next.current_nr_baselines, nr_timesteps, nr_channels);
      htodstream.memcpyHtoDAsync(d_uvw_next, job_next.uvw_ptr, sizeof_uvw);
      htodstream.memcpyHtoDAsync(d_weights_next, job_next.weights_ptr,
                                 sizeof_weights);
      htodstream.record(*inputCopied[job_id_next]);
    }

    // Wait for input to be copied
    executestream.waitEvent(*inputCopied[job_id]);

    // Launch kernel
    device.launch_average_beam(job.current_nr_baselines, nr_antennas,
                               nr_timesteps, nr_channels, nr_aterms,
                               subgrid_size, d_uvw, d_baselines, d_aterms,
                               d_aterms_offsets, d_weights, d_average_beam);
  }

  // Wait for execution to finish
  executestream.synchronize();

  // Copy result to host
  idg::Array4D<std::complex<double>> average_beam_double(subgrid_size,
                                                         subgrid_size, 4, 4);
  dtohstream.memcpyDtoHAsync(average_beam_double.data(), d_average_beam,
                             average_beam_double.bytes());

// Convert to floating-point
#pragma omp parallel for
  for (unsigned int i = 0; i < subgrid_size * subgrid_size; i++) {
    unsigned int y = i / subgrid_size;
    unsigned int x = i % subgrid_size;
    for (int ii = 0; ii < 4; ii++) {
      for (int jj = 0; jj < 4; jj++) {
        average_beam(y, x, ii, jj) += average_beam_double(y, x, ii, jj);
      }
    }
  }

  // Performance reporting
  m_report->print_total();
}

}  // end namespace cuda
}  // end namespace proxy
}  // end namespace idg
