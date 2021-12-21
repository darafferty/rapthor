// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "BufferSetImpl.h"
#include "BulkDegridderImpl.h"
#include "GridderBufferImpl.h"
#include "DegridderBufferImpl.h"
#include "common/Math.h"

#include <complex>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <csignal>

#include <omp.h>

// #if defined(HAVE_MKL)
//     #include <mkl_lapacke.h>
// #else
//     // Workaround: Prevent c-linkage of templated complex<double> in
//     lapacke.h #include <complex.h> #define lapack_complex_float    float
//     _Complex #define lapack_complex_double   double _Complex
//     // End workaround
//     #include <lapacke.h>
// #endif

#include "taper.h"
#include "idg-fft.h"
#include "npy.hpp"

#define ENABLE_VERBOSE_TIMING 0

extern "C" void cgetrf_(int* m, int* n, std::complex<float>* a, int* lda,
                        int* ipiv, int* info);

extern "C" void cgetri_(int* n, std::complex<float>* a, int* lda,
                        const int* ipiv, std::complex<float>* work, int* lwork,
                        int* info);

namespace idg {
namespace api {

BufferSet* BufferSet::create(Type architecture) {
  idg::auxiliary::print_version();
  return new BufferSetImpl(architecture);
}

uint64_t BufferSet::get_memory_per_timestep(size_t nStations, size_t nChannels,
                                            size_t nCorrelations) {
  size_t nBaselines = ((nStations - 1) * nStations) / 2;
  size_t sizeof_timestep = 0;
  sizeof_timestep +=
      auxiliary::sizeof_visibilities(nBaselines, 1, nChannels, nCorrelations);
  sizeof_timestep += auxiliary::sizeof_uvw(nBaselines, 1);
  return sizeof_timestep;
}

int nextcomposite(int n) {
  n += (n & 1);
  while (true) {
    int nn = n;
    while ((nn % 2) == 0) nn /= 2;
    while ((nn % 3) == 0) nn /= 3;
    while ((nn % 5) == 0) nn /= 5;
    if (nn == 1) return n;
    n += 2;
  }
}

BufferSetImpl::BufferSetImpl(Type architecture)
    : m_default_aterm_correction(0, 0, 0, 0),
      m_avg_aterm_correction(0, 0, 0, 0),
      m_stokes_I_only(false),
      m_nr_correlations(4),
      m_nr_polarizations(4),
      m_proxy(create_proxy(architecture)),
      m_shift(2),
      m_get_image_watch(Stopwatch::create()),
      m_set_image_watch(Stopwatch::create()),
      m_avg_beam_watch(Stopwatch::create()),
      m_plan_watch(Stopwatch::create()),
      m_gridding_watch(Stopwatch::create()),
      m_degridding_watch(Stopwatch::create()) {
  m_shift.zero();
}

BufferSetImpl::~BufferSetImpl() {
  // Free all objects allocated via the proxy before destroying the proxy.
  // CUDA free calls for those objects may not occur after destroying
  // the CUDA context, which happens when destroying the proxy.
  m_gridderbuffers.clear();
  m_degridderbuffers.clear();
  m_bulkdegridders.clear();
  m_spheroidal.free();
  m_proxy.reset();
  report_runtime();
}

std::unique_ptr<proxy::Proxy> BufferSetImpl::create_proxy(Type architecture) {
  std::unique_ptr<proxy::Proxy> proxy;

  if (architecture == Type::CPU_REFERENCE) {
#if defined(BUILD_LIB_CPU)
    proxy.reset(new proxy::cpu::Reference());
#else
    throw std::runtime_error(
        "Can not create CPU_REFERENCE proxy. idg-lib was built with "
        "BUILD_LIB_CPU=OFF");
#endif
  } else if (architecture == Type::CPU_OPTIMIZED) {
#if defined(BUILD_LIB_CPU)
    proxy.reset(new proxy::cpu::Optimized());
#else
    throw std::runtime_error(
        "Can not create CPU_OPTIMIZED proxy. idg-lib was built with "
        "BUILD_LIB_CPU=OFF");
#endif
  }
  if (architecture == Type::CUDA_GENERIC) {
#if defined(BUILD_LIB_CUDA)
    proxy.reset(new proxy::cuda::Generic());
#else
    throw std::runtime_error(
        "Can not create CUDA_GENERIC proxy. idg-lib was built with "
        "BUILD_LIB_CUDA=OFF");
#endif
  }
  if (architecture == Type::HYBRID_CUDA_CPU_OPTIMIZED) {
#if defined(BUILD_LIB_CPU) && defined(BUILD_LIB_CUDA)
    proxy.reset(new proxy::hybrid::GenericOptimized());
#else
    throw std::runtime_error(
        std::string("Can not create HYBRID_CUDA_CPU_OPTIMIZED proxy.\n") +
        std::string("For HYBRID_CUDA_CPU_OPTIMIZED idg-lib needs to be build "
                    "with BUILD_LIB_CPU=ON and BUILD_LIB_CUDA=ON\n") +
        std::string("idg-lib was built with BUILD_LIB_CPU=") +
#if defined(BUILD_LIB_CPU)
        std::string("ON")
#else
        std::string("OFF")
#endif
        + std::string(" and BUILD_LIB_CUDA=") +
#if defined(BUILD_LIB_CUDA)
        std::string("ON")
#else
        std::string("OFF")
#endif
    );
#endif
  }
  if (architecture == Type::OPENCL_GENERIC) {
#if defined(BUILD_LIB_OPENCL)
    proxy.reset(new proxy::opencl::Generic());
#else
    throw std::runtime_error(
        "Can not create OPENCL_GENERIC proxy. idg-lib was built with "
        "BUILD_LIB_OPENCL=OFF");
#endif
  }

  if (!proxy) throw std::invalid_argument("Unknown architecture type.");

  return proxy;
}

std::shared_ptr<idg::Grid> BufferSetImpl::allocate_grid() {
  m_proxy->free_grid();
  std::shared_ptr<idg::Grid> grid = m_proxy->allocate_grid(
      m_nr_w_layers, m_nr_polarizations, m_padded_size, m_padded_size);
  grid->zero();
  m_proxy->set_grid(grid);
  return grid;
}

void BufferSetImpl::init(size_t size, float cell_size, float max_w,
                         float shiftl, float shiftm, options_type& options) {
  m_average_beam.clear();

  m_stokes_I_only = false;
  if (options.count("stokes_I_only")) {
    m_stokes_I_only = options["stokes_I_only"];
  }

  if (m_stokes_I_only) {
    m_nr_correlations = 2;
    m_nr_polarizations = 1;
  } else {
    m_nr_correlations = 4;
    m_nr_polarizations = 4;
  }

  const float taper_kernel_size = 7.0;
  const float a_term_kernel_size = (options.count("a_term_kernel_size"))
                                       ? (float)options["a_term_kernel_size"]
                                       : 0.0;

  m_size = size;

  int max_threads =
      (options.count("max_threads")) ? (int)options["max_threads"] : 0;
  if (max_threads > 0) {
    omp_set_num_threads(max_threads);
  }

  int max_nr_w_layers =
      (options.count("max_nr_w_layers")) ? (int)options["max_nr_w_layers"] : 0;

  if (options.count("padded_size")) {
    m_padded_size = nextcomposite((size_t)options["padded_size"]);
  } else {
    float padding =
        (options.count("padding")) ? (double)options["padding"] : 1.20;
    m_padded_size = nextcomposite(std::ceil(m_size * padding));
  }

#ifndef NDEBUG
  std::cout << "m_padded_size: " << m_padded_size << std::endl;
#endif

  m_proxy->set_disable_wstacking(options.count("disable_wstacking") &&
                                 options["disable_wstacking"]);
  m_proxy->set_disable_wtiling(options.count("disable_wtiling") &&
                               options["disable_wtiling"]);

  //
  m_cell_size = cell_size;
  m_image_size = m_cell_size * m_padded_size;

  const float image_size_shift =
      m_image_size + 2 * std::max(std::abs(shiftl), std::abs(shiftm));

  // this cuts the w kernel approximately at the 1% level
  const float max_w_size = max_w * image_size_shift * m_image_size;

  float w_kernel_size;

  if (m_proxy->supports_wtiling()) {
    w_kernel_size = 8;
    m_w_step = 2 * w_kernel_size / (image_size_shift * m_image_size);
    m_nr_w_layers = 1;
    m_apply_wstack_correction = false;
  } else if (m_proxy->supports_wstacking()) {
    // some heuristic to set kernel size
    // square root splits the w_kernel evenly over wstack and wprojection
    // still needs a bit more thinking, and better motivation.
    // but for now does something reasonable
    w_kernel_size = std::max(8, int(std::round(2 * std::sqrt(max_w_size))));
    m_w_step = 2 * w_kernel_size / (image_size_shift * m_image_size);
    m_nr_w_layers = std::ceil(max_w / m_w_step);

    // restrict nr w layers
    if (max_nr_w_layers)
      m_nr_w_layers = std::min(max_nr_w_layers, m_nr_w_layers);
    m_w_step = max_w / m_nr_w_layers;
    w_kernel_size = 0.5 * m_w_step * image_size_shift * m_image_size;
    m_apply_wstack_correction = true;
  } else {
    w_kernel_size = max_w_size;
    m_nr_w_layers = 1;
    m_w_step = 0.0;
    m_apply_wstack_correction = false;
  }

#ifndef NDEBUG
  std::cout << "nr_w_layers: " << m_nr_w_layers << std::endl;
#endif

  m_shift(0) = shiftl;
  m_shift(1) = shiftm;

  m_kernel_size = taper_kernel_size + w_kernel_size + a_term_kernel_size;

  // reserved space in subgrid for time
  const float uv_span_time = 8.0;
  const float uv_span_frequency = 8.0;

  m_subgridsize =
      int(std::ceil((m_kernel_size + uv_span_time + uv_span_frequency) / 8.0)) *
      8;

  m_default_aterm_correction =
      Array4D<std::complex<float>>(m_subgridsize, m_subgridsize, 4, 4);
  m_default_aterm_correction.init(0.0);
  for (size_t i = 0; i < m_subgridsize; i++) {
    for (size_t j = 0; j < m_subgridsize; j++) {
      for (size_t k = 0; k < 4; k++) {
        m_default_aterm_correction(i, j, k, k) = 1.0;
      }
    }
  }

  allocate_grid();
  m_proxy->init_cache(m_subgridsize, m_cell_size, m_w_step, m_shift);

  m_taper_subgrid.resize(m_subgridsize);
  m_taper_grid.resize(m_padded_size);

  std::string tapertype;
  if (options.count("taper")) tapertype = options["taper"].as<std::string>();
  if (tapertype == "blackman-harris") {
    init_blackman_harris_1D(m_subgridsize, m_taper_subgrid.data());
    init_blackman_harris_1D(m_padded_size, m_taper_grid.data());
  } else {
    init_optimal_taper_1D(m_subgridsize, m_padded_size, m_size,
                          taper_kernel_size, m_taper_subgrid.data(),
                          m_taper_grid.data());
  }
  // Compute inverse taper
  m_inv_taper.resize(m_size);
  size_t offset = (m_padded_size - m_size) / 2;

  for (int i = 0; i < m_size; i++) {
    float y = m_taper_grid[i + offset];
    m_inv_taper[i] = 1.0 / y;
  }

  // Generate spheroidal using m_taper_subgrid.
  m_spheroidal = m_proxy->allocate_array2d<float>(m_subgridsize, m_subgridsize);
  for (size_t y = 0; y < m_subgridsize; y++) {
    for (size_t x = 0; x < m_subgridsize; x++) {
      m_spheroidal(y, x) = m_taper_subgrid[y] * m_taper_subgrid[x];
    }
  }
}

void BufferSetImpl::init_buffers(size_t bufferTimesteps,
                                 std::vector<std::vector<double>> bands,
                                 int nr_stations, float max_baseline,
                                 options_type& options,
                                 BufferSetType buffer_set_type) {
  m_gridderbuffers.clear();
  m_degridderbuffers.clear();

  m_buffer_set_type = buffer_set_type;

  for (auto band : bands) {
    BufferImpl* buffer = nullptr;
    switch (m_buffer_set_type) {
      case BufferSetType::kGridding: {
        std::unique_ptr<GridderBufferImpl> gridderbuffer(
            new GridderBufferImpl(*this, bufferTimesteps));
        gridderbuffer->set_avg_beam(
            m_average_beam.empty() ? nullptr : m_average_beam.data());
        buffer = gridderbuffer.get();
        m_gridderbuffers.push_back(std::move(gridderbuffer));
        break;
      }
      case BufferSetType::kDegridding: {
        std::unique_ptr<DegridderBufferImpl> degridderbuffer(
            new DegridderBufferImpl(*this, bufferTimesteps));
        buffer = degridderbuffer.get();
        m_degridderbuffers.push_back(std::move(degridderbuffer));
        break;
      }
      case BufferSetType::kBulkDegridding:
        m_bulkdegridders.emplace_back(
            new BulkDegridderImpl(*this, band, nr_stations));
        break;
    }

    if (buffer) {  // Perform common steps for Buffer classes.
      buffer->set_frequencies(band);
      buffer->set_stations(nr_stations);
      buffer->bake();
    }
  }
}

const BulkDegridder* BufferSetImpl::get_bulk_degridder(int i) {
  if (m_buffer_set_type != BufferSetType::kBulkDegridding) {
    throw(std::logic_error("BufferSet is not of bulk degridding type"));
  }
  return (i >= 0 && i < m_bulkdegridders.size()) ? m_bulkdegridders[i].get()
                                                 : nullptr;
}

GridderBuffer* BufferSetImpl::get_gridder(int i) {
  if (m_buffer_set_type != BufferSetType::kGridding) {
    throw(std::logic_error("BufferSet is not of gridding type"));
  }
  return m_gridderbuffers[i].get();
}

DegridderBuffer* BufferSetImpl::get_degridder(int i) {
  if (m_buffer_set_type != BufferSetType::kDegridding) {
    throw(std::logic_error("BufferSet is not of degridding type"));
  }
  return m_degridderbuffers[i].get();
}

void BufferSetImpl::set_image(const double* image, bool do_scale) {
  m_set_image_watch->Start();

  double runtime = -omp_get_wtime();
#if ENABLE_VERBOSE_TIMING
  std::cout << std::setprecision(3);
#endif

  Grid& grid = *allocate_grid();

  const int nr_w_layers = grid.get_w_dim();
  const size_t y0 = (m_padded_size - m_size) / 2;
  const size_t x0 = (m_padded_size - m_size) / 2;

  // Convert from stokes to linear into w plane 0
#if ENABLE_VERBOSE_TIMING
  std::cout << "set grid from image" << std::endl;
#endif
  double runtime_stacking = -omp_get_wtime();
  grid.zero();
#pragma omp parallel
  {
    typedef float arr_float_1D_t[m_size];
    typedef float arr_float_2D_t[m_nr_polarizations][m_size];
    const size_t size_1D = (sizeof(arr_float_1D_t) + 63) & ~size_t(63);
    const size_t size_2D = (sizeof(arr_float_2D_t) + 63) & ~size_t(63);

    arr_float_2D_t& w0_row_real __attribute__((aligned(64))) =
        *reinterpret_cast<arr_float_2D_t*>(aligned_alloc(64, size_2D));
    arr_float_2D_t& w0_row_imag __attribute__((aligned(64))) =
        *reinterpret_cast<arr_float_2D_t*>(aligned_alloc(64, size_2D));
    arr_float_2D_t& w_row_real __attribute__((aligned(64))) =
        *reinterpret_cast<arr_float_2D_t*>(aligned_alloc(64, size_2D));
    arr_float_2D_t& w_row_imag __attribute__((aligned(64))) =
        *reinterpret_cast<arr_float_2D_t*>(aligned_alloc(64, size_2D));
    arr_float_1D_t& inv_tapers __attribute__((aligned(64))) =
        *reinterpret_cast<arr_float_1D_t*>(aligned_alloc(64, size_1D));
    arr_float_1D_t& phasor_real __attribute__((aligned(64))) =
        *reinterpret_cast<arr_float_1D_t*>(aligned_alloc(64, size_1D));
    arr_float_1D_t& phasor_imag __attribute__((aligned(64))) =
        *reinterpret_cast<arr_float_1D_t*>(aligned_alloc(64, size_1D));

#pragma omp for
    for (int y = 0; y < m_size; y++) {
      memset(w0_row_real, 0, m_nr_polarizations * m_size * sizeof(float));
      memset(w0_row_imag, 0, m_nr_polarizations * m_size * sizeof(float));

      const Array3D<double> image_array(const_cast<double*>(image),
                                        m_nr_polarizations, m_size, m_size);

      // Copy row of image and convert stokes to polarizations
      if (m_stokes_I_only) {
        for (int x = 0; x < m_size; x++) {
          float scale = do_scale ? (*m_scalar_beam)[m_size * y + x] : 1.0f;
          // Stokes I
          constexpr int pol = 0;

          w0_row_real[pol][x] = image_array(0, y, x) / scale;

          // Check whether the beam response was so small (or zero) that the
          // result was non-finite. This test is done after having divided the
          // image by the beam, instead of testing the beam itself for zero,
          // because the beam can be unequal to zero and still cause an
          // overflow.
          if (!std::isfinite(w0_row_real[pol][x]) ||
              !std::isfinite(w0_row_imag[pol][x])) {
            w0_row_real[pol][x] = 0.0;
            w0_row_imag[pol][x] = 0.0;
          }
        }
      } else {
        for (int x = 0; x < m_size; x++) {
          float scale = do_scale ? (*m_scalar_beam)[m_size * y + x] : 1.0f;
          // Stokes I
          w0_row_real[0][x] = image_array(0, y, x) / scale;
          w0_row_real[3][x] = image_array(0, y, x) / scale;
          // Stokes Q
          w0_row_real[0][x] += image_array(1, y, x) / scale;
          w0_row_real[3][x] -= image_array(1, y, x) / scale;
          // Stokes U
          w0_row_real[1][x] = image_array(2, y, x) / scale;
          w0_row_real[2][x] = image_array(2, y, x) / scale;
          // Stokes V
          w0_row_imag[1][x] = -image_array(3, y, x) / scale;
          w0_row_imag[2][x] = image_array(3, y, x) / scale;

          // Check whether the beam response was so small (or zero) that the
          // result was non-finite. This test is done after having divided the
          // image by the beam, instead of testing the beam itself for zero,
          // because the beam can be unequal to zero and still cause an
          // overflow.
          for (int pol = 0; pol < m_nr_polarizations; pol++) {
            if (!std::isfinite(w0_row_real[pol][x]) ||
                !std::isfinite(w0_row_imag[pol][x])) {
              w0_row_real[pol][x] = 0.0;
              w0_row_imag[pol][x] = 0.0;
            }
          }
        }  // end for x
      }

      // Compute inverse spheroidal
      for (int x = 0; x < m_size; x++) {
        inv_tapers[x] = m_inv_taper[y] * m_inv_taper[x];
      }  // end for x

      // Copy to other w planes and multiply by w term
      for (int w = nr_w_layers - 1; w >= 0; w--) {
        // Compute current row of w-plane

        if (!m_apply_wstack_correction) {
          for (int pol = 0; pol < m_nr_polarizations; pol++) {
            for (int x = 0; x < m_size; x++) {
              w_row_real[pol][x] = w0_row_real[pol][x] * inv_tapers[x];
              w_row_imag[pol][x] = w0_row_imag[pol][x] * inv_tapers[x];
            }  // end for x
          }    // end for pol
        } else {
          // Compute phasor. Note that this code has no test coverage.
          // TODO: Test if the sign for compute_n is correct.
          for (int x = 0; x < m_size; x++) {
            const float w_offset = (w + 0.5) * m_w_step;
            const float l = (x - ((int)m_size / 2)) * m_cell_size;
            const float m = (y - ((int)m_size / 2)) * m_cell_size;
            const float n = compute_n(l, -m, m_shift.data());
            const float phase = 2 * M_PI * n * w_offset;
            phasor_real[x] = cosf(phase);
            phasor_imag[x] = sinf(phase);
          }

          // Compute current row of w-plane
          for (int pol = 0; pol < m_nr_polarizations; pol++) {
            for (int x = 0; x < m_size; x++) {
              float value_real = w0_row_real[pol][x] * inv_tapers[x];
              float value_imag = w0_row_imag[pol][x] * inv_tapers[x];
              float phasor_real_ = phasor_real[x];
              float phasor_imag_ = phasor_imag[x];
              w_row_real[pol][x] = value_real * phasor_real_;
              w_row_imag[pol][x] = value_real * phasor_imag_;
              w_row_real[pol][x] -= value_imag * phasor_imag_;
              w_row_imag[pol][x] += value_imag * phasor_real_;
            }  // end for x
          }    // end for pol
        }

        // Set m_grid
        for (int pol = 0; pol < m_nr_polarizations; pol++) {
          for (int x = 0; x < m_size; x++) {
            float value_real = w_row_real[pol][x];
            float value_imag = w_row_imag[pol][x];
            grid(w, pol, y + y0, x + x0) = {value_real, value_imag};
          }  // end for x
        }    // end for pol
      }      // end for w
    }        // end for y
    free(w0_row_real);
    free(w0_row_imag);
    free(w_row_real);
    free(w_row_imag);
    free(inv_tapers);
    free(phasor_real);
    free(phasor_imag);
  }

  runtime_stacking += omp_get_wtime();
#if ENABLE_VERBOSE_TIMING
  std::cout << "w-stacking runtime: " << runtime_stacking << std::endl;
#endif

// Fourier transform w layers
#if ENABLE_VERBOSE_TIMING
  std::cout << "fft w_layers";
#endif
  int batch = nr_w_layers * m_nr_polarizations;
  double runtime_fft = -omp_get_wtime();
  fft2f(batch, m_padded_size, m_padded_size, grid.data(0, 0, 0, 0));
  runtime_fft += omp_get_wtime();
#if ENABLE_VERBOSE_TIMING
  std::cout << ", runtime: " << runtime_fft << std::endl;
#endif

  // Report overall runtime
  runtime += omp_get_wtime();
#if ENABLE_VERBOSE_TIMING
  std::cout << "runtime " << __func__ << ": " << runtime << std::endl;
#endif

  m_set_image_watch->Pause();
}

void BufferSetImpl::write_grid(idg::Grid& grid) {
  size_t nr_w_layers = grid.get_w_dim();
  size_t nr_polarizations = grid.get_z_dim();
  size_t grid_size = grid.get_y_dim();
  assert(grid_size == grid.get_x_dim());

  std::vector<float> grid_real(nr_w_layers * nr_polarizations * grid_size *
                               grid_size * sizeof(float));
  std::vector<float> grid_imag(nr_w_layers * nr_polarizations * grid_size *
                               grid_size * sizeof(float));
  for (int w = 0; w < nr_w_layers; w++) {
#pragma omp parallel for
    for (int y = 0; y < grid_size; y++) {
      for (int x = 0; x < grid_size; x++) {
        for (int pol = 0; pol < nr_polarizations; pol++) {
          size_t idx = w * nr_polarizations * grid_size * grid_size +
                       pol * grid_size * grid_size + y * grid_size + x;
          grid_real[idx] = grid(w, pol, y, x).real();
          grid_imag[idx] = grid(w, pol, y, x).imag();
        }
      }
    }
  }
  std::cout << "writing grid to grid_real.npy and grid_imag.npy" << std::endl;
  const long unsigned leshape[] = {(long unsigned int)nr_w_layers,
                                   nr_polarizations, grid_size, grid_size};
  npy::SaveArrayAsNumpy("grid_real.npy", false, 4, leshape, grid_real);
  npy::SaveArrayAsNumpy("grid_imag.npy", false, 4, leshape, grid_imag);
}

void BufferSetImpl::get_image(double* image) {
  m_get_image_watch->Start();

  // Flush all pending operations on the grid
  const Grid& grid = *m_proxy->get_final_grid();
  int nr_polarizations = grid.get_z_dim();

  double runtime = -omp_get_wtime();
#if ENABLE_VERBOSE_TIMING
  std::cout << std::setprecision(3);
#endif

  const int nr_w_layers = grid.get_w_dim();
  const size_t y0 = (m_padded_size - m_size) / 2;
  const size_t x0 = (m_padded_size - m_size) / 2;

  // Fourier transform w layers
#if ENABLE_VERBOSE_TIMING
  std::cout << "ifft w_layers";
#endif
  int batch = nr_w_layers * nr_polarizations;
  double runtime_fft = -omp_get_wtime();
  idg::ifft2f(batch, m_padded_size, m_padded_size, grid.data(0, 0, 0, 0));
  runtime_fft += omp_get_wtime();
#if ENABLE_VERBOSE_TIMING
  std::cout << ", runtime: " << runtime_fft << std::endl;
#endif

  // Stack w layers
  double runtime_stacking = -omp_get_wtime();

#pragma omp parallel
  {
    typedef float arr_float_1D_t[m_size];
    typedef float arr_float_2D_t[m_nr_polarizations][m_size];
    const size_t size_1D = (sizeof(arr_float_1D_t) + 63) & ~size_t(63);
    const size_t size_2D = (sizeof(arr_float_2D_t) + 63) & ~size_t(63);

    arr_float_2D_t& w0_row_real __attribute__((aligned(64))) =
        *reinterpret_cast<arr_float_2D_t*>(aligned_alloc(64, size_2D));
    arr_float_2D_t& w0_row_imag __attribute__((aligned(64))) =
        *reinterpret_cast<arr_float_2D_t*>(aligned_alloc(64, size_2D));
    arr_float_2D_t& w_row_real __attribute__((aligned(64))) =
        *reinterpret_cast<arr_float_2D_t*>(aligned_alloc(64, size_2D));
    arr_float_2D_t& w_row_imag __attribute__((aligned(64))) =
        *reinterpret_cast<arr_float_2D_t*>(aligned_alloc(64, size_2D));
    arr_float_1D_t& inv_tapers __attribute__((aligned(64))) =
        *reinterpret_cast<arr_float_1D_t*>(aligned_alloc(64, size_1D));
    arr_float_1D_t& phasor_real __attribute__((aligned(64))) =
        *reinterpret_cast<arr_float_1D_t*>(aligned_alloc(64, size_1D));
    arr_float_1D_t& phasor_imag __attribute__((aligned(64))) =
        *reinterpret_cast<arr_float_1D_t*>(aligned_alloc(64, size_1D));

#pragma omp for
    for (int y = 0; y < m_size; y++) {
      Array3D<double> image_array((double*)image, m_nr_polarizations, m_size,
                                  m_size);

      // Compute inverse spheroidal
      for (int x = 0; x < m_size; x++) {
        inv_tapers[x] = m_inv_taper[y] * m_inv_taper[x];
      }

      if (!m_apply_wstack_correction) {
        // Compute current row of w-plane
        for (int pol = 0; pol < nr_polarizations; pol++) {
          for (int x = 0; x < m_size; x++) {
            auto value = grid(0, pol, y + y0, x + x0);
            w0_row_real[pol][x] = value.real() * inv_tapers[x];
            w0_row_imag[pol][x] = value.imag() * inv_tapers[x];
          }  // end for x
        }    // end for pol
      } else {
        // Compute phase. Note that this code has no test coverage.
        // TODO: Test if the sign for compute_n is correct.
        memset(w0_row_real, 0, sizeof(w0_row_real));
        memset(w0_row_imag, 0, sizeof(w0_row_imag));

        for (int w = 0; w < nr_w_layers; w++) {
          // Copy current row of w-plane
          for (int pol = 0; pol < nr_polarizations; pol++) {
            for (int x = 0; x < m_size; x++) {
              auto value = grid(w, pol, y + y0, x + x0);
              w_row_real[pol][x] = value.real();
              w_row_imag[pol][x] = value.imag();
            }  // end for pol
          }    // end for x

          // Compute phasor
          for (int x = 0; x < m_size; x++) {
            const float w_offset = (w + 0.5) * m_w_step;
            const float l = (x - ((int)m_size / 2)) * m_cell_size;
            const float m = (y - ((int)m_size / 2)) * m_cell_size;
            const float n = compute_n(l, -m, m_shift.data());
            const float phase = -2 * M_PI * n * w_offset;
            phasor_real[x] = cosf(phase);
            phasor_imag[x] = sinf(phase);
          }

          // Compute current row of w-plane
          for (int pol = 0; pol < nr_polarizations; pol++) {
            for (int x = 0; x < m_size; x++) {
              float value_real = w_row_real[pol][x] * inv_tapers[x];
              float value_imag = w_row_imag[pol][x] * inv_tapers[x];
              float phasor_real_ = phasor_real[x];
              float phasor_imag_ = phasor_imag[x];
              w_row_real[pol][x] = value_real * phasor_real_;
              w_row_real[pol][x] -= value_imag * phasor_imag_;
              if (nr_polarizations > 1) {
                // Imaginary values are only used for full polarization
                w_row_imag[pol][x] = value_real * phasor_imag_;
                w_row_imag[pol][x] += value_imag * phasor_real_;
              }
            }  // end for x
          }    // end for pol

          // Add to first w-plane
          for (int pol = 0; pol < nr_polarizations; pol++) {
            for (int x = 0; x < m_size; x++) {
              w0_row_real[pol][x] += w_row_real[pol][x];
              w0_row_imag[pol][x] += w_row_imag[pol][x];
            }  // end for x
          }    // end for pol
        }
      }  // end for w

      // Copy grid to image
      for (int x = 0; x < m_size; x++) {
        if (nr_polarizations == 4) {
          // Full polarization
          float polXX_real = w0_row_real[0][x];
          float polXY_real = w0_row_real[1][x];
          float polYX_real = w0_row_real[2][x];
          float polYY_real = w0_row_real[3][x];
          float polXY_imag = w0_row_imag[1][x];
          float polYX_imag = w0_row_imag[2][x];
          double stokesI = 0.5 * (polXX_real + polYY_real);
          double stokesQ = 0.5 * (polXX_real - polYY_real);
          double stokesU = 0.5 * (polXY_real + polYX_real);
          double stokesV = 0.5 * (-polXY_imag + polYX_imag);
          image_array(0, y, x) = stokesI;
          image_array(1, y, x) = stokesQ;
          image_array(2, y, x) = stokesU;
          image_array(3, y, x) = stokesV;
        } else if (nr_polarizations == 1) {
          // Stokes I polarization only
          double stokesI = w0_row_real[0][x];
          image_array(0, y, x) = stokesI;
        }
      }  // end for x
    }    // end for y
    free(w0_row_real);
    free(w0_row_imag);
    free(w_row_real);
    free(w_row_imag);
    free(inv_tapers);
    free(phasor_real);
    free(phasor_imag);
  }
  runtime_stacking += omp_get_wtime();
#if ENABLE_VERBOSE_TIMING
  std::cout << "w-stacking runtime: " << runtime_stacking << std::endl;
#endif

  // Free grid in proxy
  m_proxy->free_grid();

  // Report overall runtime
  runtime += omp_get_wtime();
#if ENABLE_VERBOSE_TIMING
  std::cout << "runtime " << __func__ << ": " << runtime << std::endl;
#endif
  m_get_image_watch->Pause();
}

void BufferSetImpl::finished() {
  if (m_buffer_set_type == BufferSetType::gridding) {
    for (auto& buffer : m_gridderbuffers) {
      buffer->finished();
    }
  } else {
    for (auto& buffer : m_degridderbuffers) {
      buffer->finished();
    }
  }
}

void BufferSetImpl::init_compute_avg_beam(compute_flags flag) {
  m_do_gridding = (flag != compute_flags::compute_only);
  m_average_beam =
      std::vector<std::complex<float>>(m_subgridsize * m_subgridsize * 16, 0.0);
  for (auto& buffer : m_gridderbuffers) {
    buffer->set_avg_beam(m_average_beam.data());
  }
}

void BufferSetImpl::finalize_compute_avg_beam() {
  m_avg_beam_watch->Start();

  m_matrix_inverse_beam =
      std::make_shared<std::vector<std::complex<float>>>(m_average_beam);
  m_scalar_beam = std::make_shared<std::vector<float>>(m_size * m_size);
  std::vector<std::complex<float>> scalar_beam_subgrid(
      m_subgridsize * m_subgridsize, 1.0);
  std::vector<std::complex<float>> scalar_beam_padded(
      m_padded_size * m_padded_size, 0.0);

  m_do_gridding = true;

#ifndef NDEBUG
  {
    const long unsigned leshape[] = {m_subgridsize, m_subgridsize, 4, 4};
    npy::SaveArrayAsNumpy("beam.npy", false, 4, leshape,
                          *m_matrix_inverse_beam);
  }
#endif

  for (int i = 0; i < m_subgridsize * m_subgridsize; i++) {
    //             LAPACKE_cgetrf( LAPACK_COL_MAJOR, 4, 4,
    //             (lapack_complex_float*) data, 4, ipiv); extern void cgetrf(
    //             int* m, int* n, std::complex<float>* a,
    //                     int* lda, int* ipiv, int *info );

    std::complex<float>* data = m_matrix_inverse_beam->data() + i * 16;
    int n = 4;
    int info;
    int ipiv[4];
    cgetrf_(&n, &n, data, &n, ipiv, &info);

    //             LAPACKE_cgetri( LAPACK_COL_MAJOR, 4, (lapack_complex_float*)
    //             data, 4, ipiv); extern void cgetri( int* n,
    //             std::complex<float>* a, int* lda,
    //                                 const int* ipiv, std::complex<float>*
    //                                 work, int* lwork, int *info );

    int lwork = -1;
    std::complex<float> wkopt;
    cgetri_(&n, data, &n, ipiv, &wkopt, &lwork, &info);
    lwork = int(wkopt.real());
    std::vector<std::complex<float>> work(lwork);
    cgetri_(&n, data, &n, ipiv, work.data(), &lwork, &info);

    // NOTE: there is a sign flip between the idg subgrids and the master image
    scalar_beam_subgrid[m_subgridsize * m_subgridsize - i - 1] =
        1.0 / sqrt(data[0].real() + data[3].real() + data[12].real() +
                   data[15].real());

#pragma omp simd
    for (size_t j = 0; j < 16; j++) {
      data[j] *= scalar_beam_subgrid[m_subgridsize * m_subgridsize - i - 1];
    }
  }

  // Interpolate scalar beam:
  //     1. multiply by taper
  //     2. fft
  //     3. multiply by phase gradient for half pixel shift
  //     4. zero pad
  //     5. ifft
  //     6. divide out taper and normalize

  // 1. multiply by taper
#pragma omp parallel for
  for (int i = 0; i < int(m_subgridsize); i++) {
    for (int j = 0; j < int(m_subgridsize); j++) {
      scalar_beam_subgrid[i * m_subgridsize + j] *=
          m_taper_subgrid[i] * m_taper_subgrid[j];
    }
  }

  // 2. fft
  fft2f(m_subgridsize, scalar_beam_subgrid.data());

  // 3. multiply by phase gradient for half pixel shift
#pragma omp parallel for
  for (size_t i = 0; i < m_subgridsize; i++) {
    for (size_t j = 0; j < m_subgridsize; j++) {
      float phase = -M_PI * ((float(i) + float(j)) / m_subgridsize - 1.0);

      // Compute phasor
      std::complex<float> phasor(std::cos(phase), std::sin(phase));

      scalar_beam_subgrid[i * m_subgridsize + j] *=
          phasor / float(m_subgridsize * m_subgridsize);
    }
  }

  // 4. zero pad
  {
    ptrdiff_t offset =
        (ptrdiff_t(m_padded_size) - ptrdiff_t(m_subgridsize)) / 2;
    ptrdiff_t begin_idx = std::max(-offset, ptrdiff_t(0));
    ptrdiff_t end_idx =
        std::min(ptrdiff_t(m_subgridsize), ptrdiff_t(m_padded_size) - offset);

#pragma omp parallel for
    for (ptrdiff_t i = begin_idx; i < end_idx; i++) {
      for (ptrdiff_t j = begin_idx; j < end_idx; j++) {
        scalar_beam_padded[(i + offset) * m_padded_size + (j + offset)] =
            scalar_beam_subgrid[i * m_subgridsize + j];
      }
    }
  }

  // 5. ifft
  ifft2f(m_padded_size, scalar_beam_padded.data());

  // 6. divide out taper and normalize
  {
    size_t offset = (m_padded_size - m_size) / 2;
    float x_center = m_size / 2;
    float y_center = m_size / 2;
    float center_value =
        scalar_beam_padded[(y_center + offset) * m_padded_size + x_center +
                           offset]
            .real() *
        m_inv_taper[x_center] * m_inv_taper[y_center];
    float normalization = 1.0 / center_value;
#pragma omp parallel for
    for (size_t y = 0; y < m_size; y++) {
      for (size_t x = 0; x < m_size; x++) {
        (*m_scalar_beam)[m_size * y + x] =
            scalar_beam_padded[(y + offset) * m_padded_size + x + offset]
                .real() *
            m_inv_taper[x] * m_inv_taper[y] * normalization;
      }
    }

    // normalize matrix beam as well
    for (int i = 0; i < m_subgridsize * m_subgridsize; i++) {
      std::complex<float>* data = m_matrix_inverse_beam->data() + i * 16;
#pragma omp simd
      for (size_t j = 0; j < 16; j++) {
        data[j] *= normalization;
      }
    }
  }

#if !defined(NDEBUG) || defined(WRITE_OUT_SCALAR_BEAM)
  {
    const long unsigned leshape[] = {m_size, m_size};
    npy::SaveArrayAsNumpy("scalar_beam.npy", false, 2, leshape, *m_scalar_beam);
  }
#endif

  m_avg_aterm_correction = Array4D<std::complex<float>>(
      m_matrix_inverse_beam->data(), m_subgridsize, m_subgridsize, 4, 4);
  m_proxy->set_avg_aterm_correction(m_avg_aterm_correction);

#ifndef NDEBUG
  {
    const long unsigned leshape[] = {m_subgridsize, m_subgridsize, 4, 4};
    npy::SaveArrayAsNumpy("beam_inv.npy", false, 4, leshape,
                          *m_matrix_inverse_beam);
  }
#endif

  m_avg_beam_watch->Pause();
}

void BufferSetImpl::set_matrix_inverse_beam(
    std::shared_ptr<std::vector<std::complex<float>>> matrix_inverse_beam) {
  m_matrix_inverse_beam = matrix_inverse_beam;
  m_avg_aterm_correction = Array4D<std::complex<float>>(
      m_matrix_inverse_beam->data(), m_subgridsize, m_subgridsize, 4, 4);
  m_proxy->set_avg_aterm_correction(m_avg_aterm_correction);
}

void BufferSetImpl::unset_matrix_inverse_beam() {
  m_matrix_inverse_beam.reset();
  m_avg_aterm_correction = Array4D<std::complex<float>>(0, 0, 0, 0);
  m_proxy->unset_avg_aterm_correction();
}

void BufferSetImpl::report_runtime() {
  std::clog << "avg beam:   " << m_avg_beam_watch->ToString() << std::endl;
  std::clog << "plan:       " << m_plan_watch->ToString() << std::endl;
  std::clog << "gridding:   " << m_gridding_watch->ToString() << std::endl;
  std::clog << "degridding: " << m_degridding_watch->ToString() << std::endl;
  std::clog << "set image:  " << m_set_image_watch->ToString() << std::endl;
  std::clog << "get image:  " << m_get_image_watch->ToString() << std::endl;
}

Stopwatch& BufferSetImpl::get_watch(Watch watch) const {
  switch (watch) {
    case Watch::kAvgBeam:
      return *m_avg_beam_watch;
    case Watch::kPlan:
      return *m_plan_watch;
    case Watch::kGridding:
      return *m_gridding_watch;
    case Watch::kDegridding:
      return *m_degridding_watch;
    default:
      throw std::invalid_argument("Invalid watch request");
  }
}

}  // namespace api
}  // namespace idg
