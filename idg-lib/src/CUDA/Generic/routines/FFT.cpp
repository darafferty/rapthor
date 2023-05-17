#include "../Generic.h"
#include "InstanceCUDA.h"

using namespace idg::kernel::cuda;

namespace idg {
namespace proxy {
namespace cuda {

void Generic::do_transform(DomainAtoDomainB direction) {
#if defined(DEBUG)
  std::cout << "Generic::" << __func__ << std::endl;
#endif

  // Constants
  const size_t nr_w_layers = get_grid().shape(0);
  assert(nr_w_layers == 1);
  const size_t nr_polarizations = get_grid().shape(1);
  const size_t grid_size = get_grid().shape(2);
  assert(get_grid().shape(3) == grid_size);
  const size_t sizeof_grid = get_grid().size() * sizeof(*get_grid().data());

  // Load CUDA objects
  InstanceCUDA& device = get_device(0);
  cu::Stream& stream = device.get_execute_stream();
  const cu::Context& context = device.get_context();

  // In case W-Tiling is disabled, d_grid_ is not allocated yet
  if (!m_disable_wtiling) {
    assert(!d_grid_);
    d_grid_.reset(new cu::DeviceMemory(context, sizeof_grid));
    if (m_use_unified_memory) {
      stream.memcpyHtoDAsync(*d_grid_, get_unified_grid_data(), sizeof_grid);
    } else {
      stream.memcpyHtoDAsync(*d_grid_, get_grid().data(), sizeof_grid);
    }
  }

  // Performance measurements
  get_report()->initialize(0, 0, grid_size);
  device.set_report(get_report());
  pmt::State powerStates[4];
  powerStates[0] = power_meter_->Read();
  powerStates[2] = device.measure();

  // Perform fft shift
  device.launch_fft_shift(*d_grid_, nr_polarizations, grid_size);

  // Execute fft
  device.launch_grid_fft(*d_grid_, nr_polarizations, grid_size, direction);

  // Perform fft shift and scaling
  std::complex<float> scale =
      (direction == FourierDomainToImageDomain)
          ? std::complex<float>(2.0 / (grid_size * grid_size), 0)
          : std::complex<float>(1.0, 1.0);
  device.launch_fft_shift(*d_grid_, nr_polarizations, grid_size, scale);

  // Copy grid back to the host
  if (!m_disable_wtiling) {
    if (m_use_unified_memory) {
      stream.memcpyDtoHAsync(get_unified_grid_data(), *d_grid_, sizeof_grid);
    } else {
      stream.memcpyDtoHAsync(get_grid().data(), *d_grid_, sizeof_grid);
    }
  }

  // End measurements
  stream.synchronize();
  powerStates[1] = power_meter_->Read();
  powerStates[3] = device.measure();

  // Report performance
  get_report()->update<Report::host>(powerStates[0], powerStates[1]);
  get_report()->update<Report::device>(powerStates[2], powerStates[3]);
  get_report()->print_total(nr_polarizations);

  // Free device grid
  if (!m_disable_wtiling) {
    d_grid_.reset();
  }
}

}  // end namespace cuda
}  // end namespace proxy
}  // end namespace idg