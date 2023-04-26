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
  const size_t nr_w_layers = get_grid().shape(0);
  assert(nr_w_layers == 1);
  const size_t nr_polarizations = get_grid().shape(1);
  const size_t grid_size = get_grid().shape(2);
  assert(get_grid().shape(3) == grid_size);

  // Load CUDA objects
  InstanceCUDA& device = get_device(0);
  cu::Stream& stream = device.get_execute_stream();
  cu::Context& context = device.get_context();

  // Create Array5D reference for u_grid
  unsigned int tile_size = device.get_tile_size_grid();
  unsigned int nr_tiles_1d = grid_size / tile_size;
  cu::UnifiedMemory& u_grid = get_unified_grid();
  aocommon::xt::Span<std::complex<float>, 5> grid_tiled =
      aocommon::xt::CreateSpan<std::complex<float>, 5>(
          reinterpret_cast<std::complex<float>*>(u_grid.data()),
          {nr_tiles_1d, nr_tiles_1d, nr_polarizations, tile_size, tile_size});

  // Performance measurements
  get_report()->initialize(0, 0, grid_size);
  device.set_report(get_report());
  powersensor::State powerStates[4];
  powerStates[0] = hostPowerSensor->read();
  powerStates[2] = device.measure();

  // Tile the grid backwards, this is only needed when ::transform is called
  // directly after gridding. When ::get_final_grid is called first, get_grid
  // can be used directly.
  if (m_enable_tiling && m_grid_is_tiled) {
    device.tile_grid(get_grid(), grid_tiled, false);
  }

  // Copy grid to device
  const size_t sizeof_grid = get_grid().size() * sizeof(*get_grid().data());
  cu::DeviceMemory d_grid(context, sizeof_grid);
  stream.memcpyHtoDAsync(d_grid, get_grid().data(), sizeof_grid);

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
  stream.memcpyDtoHAsync(get_grid().data(), d_grid, sizeof_grid);

  // Remember that the m_grid is not (or no longer) tiled. Since we did not
  // alter u_grid_, there is no need to perform forward tiling here.
  m_grid_is_tiled = false;

  // End measurements
  stream.synchronize();
  powerStates[1] = hostPowerSensor->read();
  powerStates[3] = device.measure();

  // Report performance
  get_report()->update<Report::host>(powerStates[0], powerStates[1]);
  get_report()->update<Report::device>(powerStates[2], powerStates[3]);
  get_report()->print_total(nr_polarizations);
}

void Unified::set_grid(aocommon::xt::Span<std::complex<float>, 4>& grid) {
  Proxy::set_grid(grid);

  const size_t grid_size = grid.shape(2);
  assert(grid.shape(3) == grid_size);

  const InstanceCUDA& device = get_device(0);
  const cu::Context& context = device.get_context();

  if (!m_enable_tiling) {
    // Allocate grid in Unified memory
    const size_t sizeof_grid = grid.size() * sizeof(*grid.data());
    cu::UnifiedMemory& u_grid = allocate_unified_grid(context, sizeof_grid);

    // Copy grid to Unified Memory
    char* first = reinterpret_cast<char*>(grid.data());
    char* result = reinterpret_cast<char*>(u_grid.data());
    std::copy_n(first, sizeof_grid, result);
  } else {
    // Allocate tiled grid in Unified memory
    const size_t nr_w_layers = grid.shape(0);
    assert(nr_w_layers == 1);
    const size_t nr_polarizations = grid.shape(1);
    const size_t grid_height = grid.shape(2);
    const size_t grid_width = grid.shape(3);
    assert(grid_height == grid_width);
    const size_t tile_size = device.get_tile_size_grid();
    const size_t nr_tiles_1d = grid_size / tile_size;
    const size_t sizeof_grid_tiled = size_t(nr_tiles_1d) * nr_tiles_1d *
                                     nr_polarizations * tile_size * tile_size *
                                     sizeof(std::complex<float>);
    cu::UnifiedMemory& u_grid =
        allocate_unified_grid(context, sizeof_grid_tiled);

    // Create 5D span for u_grid
    aocommon::xt::Span<std::complex<float>, 5> grid_tiled =
        aocommon::xt::CreateSpan<std::complex<float>, 5>(
            reinterpret_cast<std::complex<float>*>(u_grid.data()),
            {nr_tiles_1d, nr_tiles_1d, nr_polarizations, tile_size, tile_size});

    // Tile the grid
    device.tile_grid(get_grid(), grid_tiled, true);
    m_grid_is_tiled = true;
  }
}

aocommon::xt::Span<std::complex<float>, 4>& Unified::get_final_grid() {
  if (m_grid_is_tiled) {
    const InstanceCUDA& device = get_device(0);

    const size_t nr_w_layers = get_grid().shape(0);
    assert(nr_w_layers == 1);
    const size_t nr_polarizations = get_grid().shape(1);
    const size_t grid_height = get_grid().shape(2);
    const size_t grid_width = get_grid().shape(3);
    assert(grid_height == grid_width);
    const size_t tile_size = device.get_tile_size_grid();
    const size_t nr_tiles_1d = grid_height / tile_size;

    // Create 5D span for u_grid
    cu::UnifiedMemory& u_grid = get_unified_grid();
    aocommon::xt::Span<std::complex<float>, 5> grid_tiled =
        aocommon::xt::CreateSpan<std::complex<float>, 5>(
            reinterpret_cast<std::complex<float>*>(u_grid.data()),
            {nr_tiles_1d, nr_tiles_1d, nr_polarizations, tile_size, tile_size});

    // Restore the grid
    device.tile_grid(get_grid(), grid_tiled, false);
    m_grid_is_tiled = false;
  }

  return get_grid();
}

}  // namespace cuda
}  // namespace proxy
}  // namespace idg
