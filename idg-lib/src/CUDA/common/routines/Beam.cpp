#include "../CUDA.h"
#include "../InstanceCUDA.h"

using namespace idg::kernel::cuda;

namespace idg {
namespace proxy {
namespace cuda {

void CUDA::do_compute_avg_beam(
    const unsigned int nr_antennas, const unsigned int nr_channels,
    const aocommon::xt::Span<UVW<float>, 2>& uvw,
    const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
        baselines,
    const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
    const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
    const aocommon::xt::Span<float, 4>& weights,
    aocommon::xt::Span<std::complex<float>, 4>& average_beam) {
#if defined(DEBUG)
  std::cout << "CUDA::" << __func__ << std::endl;
#endif

  const size_t nr_aterms = aterm_offsets.size() - 1;
  const size_t nr_baselines = baselines.size();
  assert(uvw.shape(0) == nr_baselines);
  const size_t nr_timesteps = uvw.shape(1);
  const size_t subgrid_size = average_beam.shape(0);
  assert(average_beam.shape(1) == subgrid_size);
  const size_t nr_polarizations = 4;

  InstanceCUDA& device = get_device(0);
  cu::Context& context = device.get_context();

  // Performance reporting
  get_report()->initialize();
  device.set_report(get_report());

  // Allocate device memory
  const size_t sizeof_aterms = aterms.size() * sizeof(*aterms.data());
  const size_t sizeof_baselines = baselines.size() * sizeof(*baselines.data());
  const size_t sizeof_aterm_offsets =
      aterm_offsets.size() * sizeof(*aterm_offsets.data());
  cu::DeviceMemory d_aterms(context, sizeof_aterms);
  cu::DeviceMemory d_baselines(context, sizeof_baselines);
  cu::DeviceMemory d_aterm_offsets(context, sizeof_aterm_offsets);
  // The average beam is constructed in double-precision on the device.
  // After all baselines are processed (i.e. all contributions are added),
  // the data is copied to the host and there converted to single-precision.
  cu::DeviceMemory d_average_beam(
      context, average_beam.size() * sizeof(std::complex<double>));

  // Find jobsize given the memory requirements for uvw and weights
  int jobsize = baselines.size() / 2;
  size_t sizeof_uvw;
  size_t sizeof_weights;
  do {
    size_t bytes_free = device.get_free_memory();
    sizeof_uvw = auxiliary::sizeof_uvw(jobsize, nr_timesteps);
    sizeof_weights =
        auxiliary::sizeof_weights(jobsize, nr_timesteps, nr_channels);
    size_t bytes_required = 2 * sizeof_uvw + 2 * sizeof_weights;
    if (bytes_free > bytes_required) {
      break;
    } else {
      jobsize /= 2;
    }
  } while (jobsize > 0);

  if (jobsize == 0) {
    throw std::runtime_error(
        "Could not allocate memory for average beam kernel.");
  }

  // Allocate device memory for uvw and weights
  std::array<cu::DeviceMemory, 2> d_uvw_{
      {{context, sizeof_uvw}, {context, sizeof_uvw}}};
  std::array<cu::DeviceMemory, 2> d_weights_{
      {{context, sizeof_weights}, {context, sizeof_weights}}};

  // Initialize job data
  struct JobData {
    size_t current_nr_baselines;
    const idg::UVW<float>* uvw_ptr;
    const float* weights_ptr;
  };
  std::vector<JobData> jobs;
  for (size_t bl = 0; bl < nr_baselines; bl += jobsize) {
    JobData job;
    const size_t first_bl = bl;
    const size_t last_bl = std::min(bl + jobsize, nr_baselines);
    job.current_nr_baselines = last_bl - first_bl;
    job.uvw_ptr = &uvw(first_bl, 0);
    job.weights_ptr = &weights(first_bl, 0, 0, 0);
    if (job.current_nr_baselines > 0) {
      jobs.push_back(job);
    }
  }

  // Events
  std::vector<std::unique_ptr<cu::Event>> inputCopied;
  for (unsigned bl = 0; bl < nr_baselines; bl += jobsize) {
    inputCopied.emplace_back(new cu::Event(context));
  }

  // Load streams
  cu::Stream& htodstream = device.get_htod_stream();
  cu::Stream& dtohstream = device.get_dtoh_stream();
  cu::Stream& executestream = device.get_execute_stream();

  // Copy static data
  htodstream.memcpyHtoDAsync(d_aterms, aterms.data(), sizeof_aterms);
  htodstream.memcpyHtoDAsync(d_baselines, baselines.data(), sizeof_baselines);
  htodstream.memcpyHtoDAsync(d_aterm_offsets, aterm_offsets.data(),
                             sizeof_aterm_offsets);

  // Initialize average beam
  d_average_beam.zero(htodstream);

  for (unsigned int job_id = 0; job_id < jobs.size(); job_id++) {
    // Id for double-buffering
    unsigned local_id = job_id % 2;
    unsigned job_id_next = job_id + 1;
    unsigned local_id_next = (local_id + 1) % 2;

    auto& job = jobs[job_id];
    cu::DeviceMemory& d_uvw = d_uvw_[local_id];
    cu::DeviceMemory& d_weights = d_weights_[local_id];

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
      cu::DeviceMemory& d_uvw_next = d_uvw_[local_id_next];
      cu::DeviceMemory& d_weights_next = d_weights_[local_id_next];
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
                               d_aterm_offsets, d_weights, d_average_beam);
  }

  // Wait for execution to finish
  executestream.synchronize();

  // Copy result to host
  Tensor<std::complex<double>, 4> average_beam_double =
      allocate_tensor<std::complex<double>, 4>(
          {subgrid_size, subgrid_size, 4, 4});
  dtohstream.memcpyDtoH(average_beam_double.Span().data(), d_average_beam,
                        average_beam_double.Span().size() *
                            sizeof(*average_beam_double.Span().data()));

// Convert to floating-point
#pragma omp parallel for
  for (size_t i = 0; i < subgrid_size * subgrid_size; i++) {
    const size_t y = i / subgrid_size;
    const size_t x = i % subgrid_size;
    for (size_t ii = 0; ii < 4; ii++) {
      for (size_t jj = 0; jj < 4; jj++) {
        average_beam(y, x, ii, jj) += average_beam_double.Span()(y, x, ii, jj);
      }
    }
  }

  // Performance reporting
  get_report()->print_total(nr_polarizations);
}

}  // end namespace cuda
}  // end namespace proxy
}  // end namespace idg
