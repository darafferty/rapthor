// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>  // max_element

#include "Unified.h"
#include "InstanceCUDA.h"

using namespace std;
using namespace idg::kernel::cuda;

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

  // Performance measurements
  get_report()->initialize(0, 0, grid_size);
  device.set_report(get_report());
  pmt::State powerStates[4];
  powerStates[0] = power_meter_->Read();
  powerStates[2] = device.measure();

  // Tile the grid backwards, this is only needed when ::transform is called
  // directly after gridding. When ::get_final_grid is called first, get_grid
  // can be used directly.
  if (m_enable_tiling && m_grid_is_tiled) {
    device.tile_grid(get_grid(), unified_grid_tiled_.Span(), false);
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

  // Remember that grid is not tiled now.
  m_grid_is_tiled = false;

  // End measurements
  stream.synchronize();
  powerStates[1] = power_meter_->Read();
  powerStates[3] = device.measure();

  // Report performance
  get_report()->update<Report::host>(powerStates[0], powerStates[1]);
  get_report()->update<Report::device>(powerStates[2], powerStates[3]);
  get_report()->print_total(nr_polarizations);
}

void Unified::set_grid(aocommon::xt::Span<std::complex<float>, 4>& grid) {
  Proxy::set_grid(grid);

  const size_t nr_w_layers = grid.shape(0);
  assert(nr_w_layers == 1);
  const size_t nr_polarizations = grid.shape(1);
  const size_t grid_height = grid.shape(2);
  const size_t grid_width = grid.shape(3);
  assert(grid_height == grid_width);
  const size_t grid_size = grid_height;

  const InstanceCUDA& device = get_device(0);
  const cu::Context& context = device.get_context();

  if (!m_enable_tiling) {
    const size_t sizeof_grid = grid.size() * sizeof(*grid.data());
    unified_grid_tiled_ = Tensor<std::complex<float>, 5>(
        std::make_unique<cu::UnifiedMemory>(context, sizeof_grid),
        {1, 1, nr_polarizations, grid_size, grid_size});

    std::copy_n(grid.data(), grid.size(), unified_grid_tiled_.Span().data());
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
    unified_grid_tiled_ = Tensor<std::complex<float>, 5>(
        std::make_unique<cu::UnifiedMemory>(context, sizeof_grid_tiled),
        {nr_tiles_1d, nr_tiles_1d, nr_polarizations, tile_size, tile_size});

    // Tile the grid
    device.tile_grid(get_grid(), unified_grid_tiled_.Span(), true);
    m_grid_is_tiled = true;
  }
}

aocommon::xt::Span<std::complex<float>, 4>& Unified::get_final_grid() {
  if (m_grid_is_tiled) {
    // Restore the grid
    const InstanceCUDA& device = get_device(0);
    device.tile_grid(get_grid(), unified_grid_tiled_.Span(), false);
    m_grid_is_tiled = false;
  }

  return get_grid();
}

}  // namespace cuda
}  // namespace proxy
}  // namespace idg
