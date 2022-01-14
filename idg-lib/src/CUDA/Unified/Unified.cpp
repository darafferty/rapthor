// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>  // max_element

#include "Unified.h"
#include "InstanceCUDA.h"

using namespace std;
using namespace idg::kernel::cuda;
using namespace powersensor;

namespace idg {
namespace proxy {
namespace cuda {

// Constructor
Unified::Unified(ProxyInfo info) : Generic(info) {
#if defined(DEBUG)
  cout << "Unified::" << __func__ << endl;
#endif

  // Increase the fraction of reserved memory
  set_fraction_reserved(0.4);

  // Enable unified memory
  enable_unified_memory();
}

// Destructor
Unified::~Unified() {
#if defined(DEBUG)
  std::cout << "Unified::" << __func__ << std::endl;
#endif
}

void Unified::do_transform(idg::DomainAtoDomainB direction) {
#if defined(DEBUG)
  std::cout << "Unified::" << __func__ << std::endl;
#endif

  // Constants
  unsigned int grid_size = m_grid->get_x_dim();
  unsigned int nr_w_layers = m_grid->get_w_dim();
  assert(nr_w_layers == 1);
  unsigned int nr_polarizations = m_grid->get_z_dim();

  // Load CUDA objects
  InstanceCUDA& device = get_device(0);
  cu::Stream& stream = device.get_execute_stream();
  cu::Context& context = device.get_context();

  // Create Array5D reference for u_grid
  unsigned int tile_size = device.get_tile_size_grid();
  unsigned int nr_tiles_1d = grid_size / tile_size;
  cu::UnifiedMemory& u_grid = get_unified_grid();
  idg::Array5D<std::complex<float>> grid_tiled(
      static_cast<std::complex<float>*>(u_grid.data()), nr_tiles_1d,
      nr_tiles_1d, nr_polarizations, tile_size, tile_size);

  // Performance measurements
  m_report->initialize(0, 0, grid_size);
  device.set_report(m_report);
  powersensor::State powerStates[4];
  powerStates[0] = hostPowerSensor->read();
  powerStates[2] = device.measure();

  // Tile the grid backwards, this is only needed when ::transform is called
  // directly after gridding. When ::get_final_grid is called first, m_grid
  // can be used directly.
  if (m_enable_tiling && m_grid_is_tiled) {
    device.tile_backward(nr_polarizations, grid_size, tile_size, grid_tiled,
                         *m_grid);
  }

  // Copy grid to device
  cu::DeviceMemory d_grid(context, m_grid->bytes());
  stream.memcpyHtoDAsync(d_grid, m_grid->data(), m_grid->bytes());

  // Perform fft shift
  device.launch_fft_shift(d_grid, nr_polarizations, grid_size);

  // Execute fft
  device.launch_grid_fft(d_grid, nr_polarizations, grid_size, direction);

  // Perform fft shift and scaling
  std::complex<float> scale =
      (direction == FourierDomainToImageDomain)
          ? std::complex<float>(2.0 / (grid_size * grid_size), 0)
          : std::complex<float>(1.0, 1.0);
  device.launch_fft_shift(d_grid, nr_polarizations, grid_size, scale);

  // Copy grid back to the host
  stream.memcpyDtoHAsync(m_grid->data(), d_grid, m_grid->bytes());

  // Remember that the m_grid is not (or no longer) tiled. Since we did not
  // alter u_grid_, there is no need to perform forward tiling here.
  m_grid_is_tiled = false;

  // End measurements
  stream.synchronize();
  powerStates[1] = hostPowerSensor->read();
  powerStates[3] = device.measure();

  // Report performance
  m_report->update<Report::host>(powerStates[0], powerStates[1]);
  m_report->update<Report::device>(powerStates[2], powerStates[3]);
  m_report->print_total(nr_polarizations);
}

void Unified::set_grid(std::shared_ptr<Grid> grid) {
  m_grid = grid;

  const InstanceCUDA& device = get_device(0);
  const cu::Context& context = device.get_context();

  if (!m_enable_tiling) {
    // Allocate grid in Unified memory
    cu::UnifiedMemory& u_grid = allocate_unified_grid(context, grid->bytes());

    // Copy grid to Unified Memory
    char* first = reinterpret_cast<char*>(grid->data());
    char* result = reinterpret_cast<char*>(u_grid.data());
    std::copy_n(first, grid->bytes(), result);
  } else {
    // Allocate tiled grid in Unified memory
    int nr_w_layers = grid->get_w_dim();
    assert(nr_w_layers == 1);
    int nr_polarizations = grid->get_z_dim();
    int grid_height = grid->get_y_dim();
    int grid_width = grid->get_x_dim();
    assert(grid_height == grid_width);
    int grid_size = grid_width;
    int tile_size = device.get_tile_size_grid();
    int nr_tiles_1d = grid_size / tile_size;
    size_t sizeof_grid_tiled = size_t(nr_tiles_1d) * nr_tiles_1d *
                               nr_polarizations * tile_size * tile_size *
                               sizeof(std::complex<float>);
    cu::UnifiedMemory& u_grid =
        allocate_unified_grid(context, sizeof_grid_tiled);

    // Forward tiling
    idg::Array5D<std::complex<float>> grid_tiled(
        static_cast<std::complex<float>*>(u_grid.data()), nr_tiles_1d,
        nr_tiles_1d, nr_polarizations, tile_size, tile_size);
    device.tile_forward(nr_polarizations, grid_size, tile_size, *grid,
                        grid_tiled);
    m_grid_is_tiled = true;
  }
}

std::shared_ptr<Grid> Unified::get_final_grid() {
  if (m_grid_is_tiled) {
    const InstanceCUDA& device = get_device(0);

    // Create Array5D reference for u_grid
    int nr_polarizations = m_grid->get_z_dim();
    int grid_size = m_grid->get_x_dim();
    int tile_size = device.get_tile_size_grid();
    int nr_tiles_1d = grid_size / tile_size;
    cu::UnifiedMemory& u_grid = get_unified_grid();
    idg::Array5D<std::complex<float>> grid_tiled(
        static_cast<std::complex<float>*>(u_grid.data()), nr_tiles_1d,
        nr_tiles_1d, nr_polarizations, tile_size, tile_size);

    // Restore the grid
    device.tile_backward(nr_polarizations, grid_size, tile_size, grid_tiled,
                         *m_grid);
    m_grid_is_tiled = false;
  }

  return m_grid;
}

}  // namespace cuda
}  // namespace proxy
}  // namespace idg
