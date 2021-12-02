#include "../Generic.h"
#include "InstanceCUDA.h"

using namespace idg::kernel::cuda;

namespace idg {
namespace proxy {
namespace cuda {

void Generic::do_transform(DomainAtoDomainB direction) {
#if defined(DEBUG)
  std::cout << "CUDA::" << __func__ << std::endl;
#endif

  check_grid();

  // Constants
  unsigned int grid_size = m_grid->get_x_dim();
  unsigned int nr_w_layers = m_grid->get_w_dim();
  assert(nr_w_layers == 1);
  unsigned int nr_polarizations = m_grid->get_z_dim();

  // Load device
  InstanceCUDA& device = get_device(0);

  // Initialize
  cu::Stream& stream = device.get_execute_stream();

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

  // End measurements
  stream.synchronize();
  powerStates[1] = hostPowerSensor->read();
  powerStates[3] = device.measure();

  // Report performance
  m_report->update<Report::host>(powerStates[0], powerStates[1]);
  m_report->update<Report::device>(powerStates[2], powerStates[3]);
  m_report->print_total(nr_polarizations);
}

}  // end namespace cuda
}  // end namespace proxy
}  // end namespace idg