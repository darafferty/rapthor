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
  unsigned int grid_size = m_grid->get_x_dim();
  unsigned int nr_w_layers = m_grid->get_w_dim();
  assert(nr_w_layers == 1);
  unsigned int nr_polarizations = m_grid->get_z_dim();

  // Load CUDA objects
  InstanceCUDA& device = get_device(0);
  cu::Stream& stream = device.get_execute_stream();
  const cu::Context& context = device.get_context();

  // In case W-Tiling is disabled, d_grid_ is not allocated yet
  if (!m_disable_wtiling) {
    assert(!d_grid_);
    d_grid_.reset(new cu::DeviceMemory(context, m_grid->bytes()));
    if (m_use_unified_memory) {
      cu::UnifiedMemory& u_grid = get_unified_grid();
      stream.memcpyHtoDAsync(*d_grid_, u_grid.data(), u_grid.size());
    } else {
      stream.memcpyHtoDAsync(*d_grid_, m_grid->data(), m_grid->bytes());
    }
  }

  // Performance measurements
  m_report->initialize(0, 0, grid_size);
  device.set_report(m_report);
  powersensor::State powerStates[4];
  powerStates[0] = hostPowerSensor->read();
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
      cu::UnifiedMemory& u_grid = get_unified_grid();
      stream.memcpyDtoHAsync(u_grid.data(), *d_grid_, u_grid.size());
    } else {
      stream.memcpyDtoHAsync(m_grid->data(), *d_grid_, m_grid->bytes());
    }
  }

  // End measurements
  stream.synchronize();
  powerStates[1] = hostPowerSensor->read();
  powerStates[3] = device.measure();

  // Report performance
  m_report->update<Report::host>(powerStates[0], powerStates[1]);
  m_report->update<Report::device>(powerStates[2], powerStates[3]);
  m_report->print_total(nr_polarizations);

  // Free device grid
  if (!m_disable_wtiling) {
    d_grid_.reset();
  }
}

}  // end namespace cuda
}  // end namespace proxy
}  // end namespace idg