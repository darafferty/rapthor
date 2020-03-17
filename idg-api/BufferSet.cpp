#include "BufferSetImpl.h"
#include "GridderBufferImpl.h"
#include "DegridderBufferImpl.h"

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
//     // Workaround: Prevent c-linkage of templated complex<double> in lapacke.h
//     #include <complex.h>
//     #define lapack_complex_float    float _Complex
//     #define lapack_complex_double   double _Complex
//     // End workaround
//     #include <lapacke.h>
// #endif

#include "taper.h"
#include "idg-fft.h"
#include "npy.hpp"

#define ENABLE_VERBOSE_TIMING 0

extern "C" void cgetrf_( int* m, int* n, std::complex<float>* a,
                    int* lda, int* ipiv, int *info );

extern "C" void cgetri_( int* n, std::complex<float>* a, int* lda,
                    const int* ipiv, std::complex<float>* work,
                    int* lwork, int *info );

namespace idg {
namespace api {

    BufferSet* BufferSet::create(Type architecture)
    {
        idg::auxiliary::print_version();
        return new BufferSetImpl(architecture);
    }

    uint64_t BufferSet::get_memory_per_timestep(size_t nStations, size_t nChannels)
    {
        size_t nBaselines = ((nStations - 1) * nStations) / 2;
        size_t sizeof_timestep = 0;
        sizeof_timestep += auxiliary::sizeof_visibilities(nBaselines, 1, nChannels);
        sizeof_timestep += auxiliary::sizeof_uvw(nBaselines, 1);
        return sizeof_timestep;
    }

    int nextcomposite(int n)
    {
        n += (n & 1);
        while (true)
        {
            int nn = n;
            while ((nn % 2) == 0) nn /= 2;
            while ((nn % 3) == 0) nn /= 3;
            while ((nn % 5) == 0) nn /= 5;
            if (nn == 1) return n;
            n += 2;
        }
    }

    BufferSetImpl::BufferSetImpl(
        Type architecture) :
        m_architecture(architecture),
        m_default_aterm_correction(0,0,0,0),
        m_avg_aterm_correction(0,0,0,0),
        m_grid(new Grid(0, 0, 0, 0)),
        m_proxy(create_proxy()),
        m_get_image_watch(Stopwatch::create()),
        m_set_image_watch(Stopwatch::create()),
        m_avg_beam_watch(Stopwatch::create()),
        m_plan_watch(Stopwatch::create()),
        m_gridding_watch(Stopwatch::create()),
        m_degridding_watch(Stopwatch::create())
    {}

    BufferSetImpl::~BufferSetImpl() {
        m_gridderbuffers.clear();
        m_degridderbuffers.clear();
        if (m_proxy) { delete m_proxy; }
        report_runtime();
    }

    proxy::Proxy* BufferSetImpl::create_proxy()
    {
        proxy::Proxy* proxy;
        int nr_correlations = 4;

        if (m_architecture == Type::CPU_REFERENCE) {
            #if defined(BUILD_LIB_CPU)
                proxy = new proxy::cpu::Reference();
            #else
                throw std::runtime_error("Can not create CPU_REFERENCE proxy. idg-lib was built with BUILD_LIB_CPU=OFF");
            #endif
        } else if (m_architecture == Type::CPU_OPTIMIZED) {
            #if defined(BUILD_LIB_CPU)
                proxy = new proxy::cpu::Optimized();
            #else
                throw std::runtime_error("Can not create CPU_OPTIMIZED proxy. idg-lib was built with BUILD_LIB_CPU=OFF");
            #endif
        }
        if (m_architecture == Type::CUDA_GENERIC) {
            #if defined(BUILD_LIB_CUDA)
                proxy = new proxy::cuda::Generic();
            #else
                throw std::runtime_error("Can not create CUDA_GENERIC proxy. idg-lib was built with BUILD_LIB_CUDA=OFF");
            #endif
        }
        if (m_architecture == Type::HYBRID_CUDA_CPU_OPTIMIZED) {
            #if defined(BUILD_LIB_CPU) && defined(BUILD_LIB_CUDA)
                proxy = new proxy::hybrid::GenericOptimized();
            #else
                throw std::runtime_error(
                    std::string("Can not create HYBRID_CUDA_CPU_OPTIMIZED proxy.\n") +
                    std::string("For HYBRID_CUDA_CPU_OPTIMIZED idg-lib needs to be build with BUILD_LIB_CPU=ON and BUILD_LIB_CUDA=ON\n") +
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
        if (m_architecture == Type::OPENCL_GENERIC) {
            #if defined(BUILD_LIB_OPENCL)
                proxy = new proxy::opencl::Generic();
            #else
                throw std::runtime_error("Can not create OPENCL_GENERIC proxy. idg-lib was built with BUILD_LIB_OPENCL=OFF");
            #endif
        }

        if (proxy == nullptr)
            throw std::invalid_argument("Unknown architecture type.");

        return proxy;

    }

    void BufferSetImpl::init(
        size_t size,
        float cell_size,
        float max_w,
        float shiftl, float shiftm, float shiftp,
        options_type &options)
    {
        const float taper_kernel_size = 7.0;
        const float a_term_kernel_size = (options.count("a_term_kernel_size")) ? (float)options["a_term_kernel_size"] : 0.0;

        m_size = size;

        int max_threads = (options.count("max_threads")) ? (int)options["max_threads"] : 0;
        if (max_threads > 0)
        {
            omp_set_num_threads(max_threads);
        }

        int max_nr_w_layers = (options.count("max_nr_w_layers")) ? (int)options["max_nr_w_layers"] : 0;

        if (options.count("padded_size"))
        {
            m_padded_size = nextcomposite((size_t)options["padded_size"]);
        }
        else
        {
            float padding = (options.count("padding")) ? (double)options["padding"] : 1.20;
            m_padded_size = nextcomposite(std::ceil(m_size * padding));
        }

#ifndef NDEBUG
        std::cout << "m_padded_size: " << m_padded_size << std::endl;
#endif
        //
        m_cell_size = cell_size;
        m_image_size = m_cell_size * m_padded_size;

        // this cuts the w kernel approximately at the 1% level
        const float max_w_size = max_w * m_image_size * m_image_size;

        // some heuristic to set kernel size
        // square root splits the w_kernel evenly over wstack and wprojection
        // still needs a bit more thinking, and better motivation.
        // but for now does something reasonable
        float w_kernel_size = std::max(8, int(std::round(2*std::sqrt(max_w_size))));

        m_w_step = 2 * w_kernel_size / (m_image_size * m_image_size);

        int nr_w_layers = std::ceil(max_w / m_w_step);

        //restrict nr w layers
        if (max_nr_w_layers) nr_w_layers = std::min(max_nr_w_layers, nr_w_layers);

#ifndef NDEBUG
        std::cout << "nr_w_layers: " << nr_w_layers << std::endl;
#endif

        m_w_step = max_w / nr_w_layers;
        m_shift[0] = shiftl;
        m_shift[1] = shiftm;
        m_shift[2] = shiftp;
        w_kernel_size = 0.5*m_w_step * m_image_size * m_image_size;

        // DEBUG no w-stacking
//         w_kernel_size = max_w_size;
//         nr_w_layers = 1;
//         m_w_step = 0.0;

        m_kernel_size = taper_kernel_size + w_kernel_size + a_term_kernel_size;

        // reserved space in subgrid for time
        m_uv_span_time = 8.0;

        m_uv_span_frequency = 8.0;

        m_subgridsize = int(std::ceil((m_kernel_size + m_uv_span_time + m_uv_span_frequency)/8.0))*8;

        m_default_aterm_correction = Array4D<std::complex<float>>(m_subgridsize, m_subgridsize, 4, 4);
        m_default_aterm_correction.init(0.0);
        for (size_t i = 0; i < m_subgridsize; i++)
        {
            for (size_t j = 0; j < m_subgridsize; j++)
            {
                for (size_t k = 0; k < 4; k++)
                {
                    m_default_aterm_correction(i,j,k,k) = 1.0;
                }
            }
        }

        m_grid.reset(new Grid(nr_w_layers,4,m_padded_size,m_padded_size));
        m_grid->zero();

        m_taper_subgrid.resize(m_subgridsize);
        m_taper_grid.resize(m_padded_size);

        std::string tapertype;
        if(options.count("taper"))
          tapertype = options["taper"].as<std::string>();
        if(tapertype == "blackman-harris")
        {
          init_blackman_harris_1D(m_subgridsize, m_taper_subgrid.data());
          init_blackman_harris_1D(m_padded_size, m_taper_grid.data());
        }
        else {
          init_optimal_taper_1D(m_subgridsize, m_padded_size, m_size, taper_kernel_size, m_taper_subgrid.data(), m_taper_grid.data());
        }
        // Compute inverse taper
        m_inv_taper.resize(m_size);
        size_t offset = (m_padded_size-m_size)/2;

        for (int i = 0; i < m_size; i++)
        {
            float y = m_taper_grid[i+offset];
            m_inv_taper[i] = 1.0/y;
        }
    }


    void BufferSetImpl::init_buffers(
            size_t bufferTimesteps,
            std::vector<std::vector<double>> bands,
            int nr_stations,
            float max_baseline,
            options_type &options,
            BufferSetType buffer_set_type)
    {
        m_gridderbuffers.resize(0);
        m_degridderbuffers.resize(0);

        m_buffer_set_type = buffer_set_type;
        std::vector<float> taper;
        taper.resize(m_subgridsize * m_subgridsize);
        for(int i = 0; i < int(m_subgridsize); i++)
            for(int j = 0; j < int(m_subgridsize); j++)
                taper[i*m_subgridsize + j] = m_taper_subgrid[i] * m_taper_subgrid[j];

        for (auto band : bands )
        {
            BufferImpl *buffer;
            if (m_buffer_set_type == BufferSetType::gridding)
            {
                GridderBufferImpl *gridderbuffer = new GridderBufferImpl(this, m_proxy, bufferTimesteps);
                m_gridderbuffers.push_back(std::unique_ptr<GridderBuffer>(gridderbuffer));
                buffer = gridderbuffer;
            }
            else
            {
                DegridderBufferImpl *degridderbuffer = new DegridderBufferImpl(this, m_proxy, bufferTimesteps);
                m_degridderbuffers.push_back(std::unique_ptr<DegridderBuffer>(degridderbuffer));
                buffer = degridderbuffer;
            }


            // TODO: maybe just give the Buffers a pointer to their parent BufferSet and make them friends of BufferSet
            // so this list of parameters does not need to passed along to the Buffers

            buffer->set_subgrid_size(m_subgridsize);

            buffer->set_frequencies(band);
            buffer->set_stations(nr_stations);
            buffer->set_cell_size(m_cell_size, m_cell_size);
            buffer->set_w_step(m_w_step);
            buffer->set_shift(m_shift);
            buffer->set_kernel_size(m_kernel_size);
            buffer->set_spheroidal(m_subgridsize, m_subgridsize, taper.data());
            buffer->set_grid(m_grid);
            buffer->set_max_baseline(max_baseline);
            buffer->set_uv_span_frequency(m_uv_span_frequency);
            buffer->bake();
        }
    }


    GridderBuffer* BufferSetImpl::get_gridder(int i)
    {
        if (m_buffer_set_type != BufferSetType::gridding)
        {
            throw(std::logic_error("BufferSet is not of gridding type"));
        }
        return m_gridderbuffers[i].get();
    }

    DegridderBuffer* BufferSetImpl::get_degridder(int i)
    {
        if (m_buffer_set_type != BufferSetType::degridding)
        {
            throw(std::logic_error("BufferSet is not of degridding type"));
        }
        return m_degridderbuffers[i].get();
    }

    // Copy of compute_n in Common/Math.h
    inline float compute_n(
        float l,
        float m,
        const float* __restrict__ shift)
    {
        const float lc = l + shift[0];
        const float mc = m + shift[1];
        const float tmp = (lc * lc) + (mc * mc);
        return tmp > 1.0 ? 1.0 : tmp / (1.0f + sqrtf(1.0f - tmp)) + shift[2];

        // evaluate n = 1.0f - sqrt(1.0 - (l * l) - (m * m));
        // accurately for small values of l and m
        //return tmp > 1.0 ? 1.0 : tmp / (1.0f + sqrtf(1.0f - tmp));
    }

    void BufferSetImpl::set_image(const double* image, bool do_scale)
    {
        m_set_image_watch->Start();

        double runtime = -omp_get_wtime();
#if ENABLE_VERBOSE_TIMING
        std::cout << std::setprecision(3);
#endif

        const int nr_w_layers = m_grid->get_w_dim();
        const size_t y0 = (m_padded_size-m_size)/2;
        const size_t x0 = (m_padded_size-m_size)/2;

        // Convert from stokes to linear into w plane 0
#if ENABLE_VERBOSE_TIMING
        std::cout << "set grid from image" << std::endl;
#endif
        double runtime_copy = -omp_get_wtime();
        m_grid->zero();
        if (do_scale)
        {
#if ENABLE_VERBOSE_TIMING
            std::cout << "scale: " << (*m_scalar_beam)[0] << std::endl;
#endif
            #pragma omp parallel for
            for (int y = 0; y < m_size; y++) {
                for (int x = 0; x < m_size; x++) {
                    // Stokes I
                    (*m_grid)(0,0,y+y0,x+x0) = image[m_size*y+x]/(*m_scalar_beam)[m_size*y+x];
                    (*m_grid)(0,3,y+y0,x+x0) = image[m_size*y+x]/(*m_scalar_beam)[m_size*y+x];
                    // Stokes Q
                    (*m_grid)(0,0,y+y0,x+x0) += image[m_size*m_size + m_size*y+x]/(*m_scalar_beam)[m_size*y+x];
                    (*m_grid)(0,3,y+y0,x+x0) -= image[m_size*m_size + m_size*y+x]/(*m_scalar_beam)[m_size*y+x];
                    // Stokes U
                    (*m_grid)(0,1,y+y0,x+x0) = image[2*m_size*m_size + m_size*y+x]/(*m_scalar_beam)[m_size*y+x];
                    (*m_grid)(0,2,y+y0,x+x0) = image[2*m_size*m_size + m_size*y+x]/(*m_scalar_beam)[m_size*y+x];
                    // Stokes V
                    (*m_grid)(0,1,y+y0,x+x0).imag(-image[3*m_size*m_size + m_size*y+x]/(*m_scalar_beam)[m_size*y+x]);
                    (*m_grid)(0,2,y+y0,x+x0).imag( image[3*m_size*m_size + m_size*y+x]/(*m_scalar_beam)[m_size*y+x]);
                } // end for x
            } // end for y
        }
        else
        {
            #pragma omp parallel for
            for (int y = 0; y < m_size; y++) {
                for (int x = 0; x < m_size; x++) {
                    // Stokes I
                    (*m_grid)(0,0,y+y0,x+x0) = image[m_size*y+x];
                    (*m_grid)(0,3,y+y0,x+x0) = image[m_size*y+x];
                    // Stokes Q
                    (*m_grid)(0,0,y+y0,x+x0) += image[m_size*m_size + m_size*y+x];
                    (*m_grid)(0,3,y+y0,x+x0) -= image[m_size*m_size + m_size*y+x];
                    // Stokes U
                    (*m_grid)(0,1,y+y0,x+x0) = image[2*m_size*m_size + m_size*y+x];
                    (*m_grid)(0,2,y+y0,x+x0) = image[2*m_size*m_size + m_size*y+x];
                    // Stokes V
                    (*m_grid)(0,1,y+y0,x+x0).imag(-image[3*m_size*m_size + m_size*y+x]);
                    (*m_grid)(0,2,y+y0,x+x0).imag( image[3*m_size*m_size + m_size*y+x]);
                } // end for x
            } // end for y
        }
        runtime_copy += omp_get_wtime();
#if ENABLE_VERBOSE_TIMING
        std::cout << "runtime:" << runtime_copy << std::endl;
#endif

        // Copy to other w planes and multiply by w term
        double runtime_stacking = -omp_get_wtime();
        for (int w = nr_w_layers - 1; w >= 0; w--) {
#if ENABLE_VERBOSE_TIMING
            std::cout << "unstacking w_layer: " << w+1 << "/" << nr_w_layers << std::endl;
#endif

            #pragma omp parallel for
            for(int y = 0; y < m_size; y++) {
                for(int x = 0; x < m_size; x++) {
                    // Compute phase
                    const float w_offset = (w+0.5)*m_w_step;
                    const float l = (x-((int)m_size/2)) * m_cell_size;
                    const float m = (y-((int)m_size/2)) * m_cell_size;
                    // evaluate n = 1.0f - sqrt(1.0 - (l * l) - (m * m));
                    // accurately for small values of l and m
                    const float n = compute_n(l, -m, m_shift);
                    //const float tmp = (l * l) + (m * m);
                    //const float n = tmp > 1.0 ? 1.0 : tmp / (1.0f + sqrtf(1.0f - tmp));
                    float phase = 2*M_PI*n*w_offset;

                    // Compute phasor
                    std::complex<float> phasor(std::cos(phase), std::sin(phase));

                    // Compute inverse spheroidal
                    float inv_taper = m_inv_taper[y] * m_inv_taper[x];

                    // Set to current w-plane
                    #pragma unroll
                    for (int pol = 0; pol < 4; pol++) {
                        (*m_grid)(w, pol, y+y0, x+x0) = (*m_grid)(0, pol, y+y0, x+x0) * inv_taper * phasor;
                    }
                } // end for x
            } // end for y
        } // end for w
        runtime_stacking += omp_get_wtime();
#if ENABLE_VERBOSE_TIMING
        std::cout << "w-stacking runtime: " << runtime_stacking << std::endl;
#endif

        // Fourier transform w layers
#if ENABLE_VERBOSE_TIMING
        std::cout << "fft w_layers";
#endif
        int batch = nr_w_layers * 4;
        double runtime_fft = -omp_get_wtime();
        fft2f(batch, m_padded_size, m_padded_size, m_grid->data(0,0,0,0));
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
    };

    void BufferSetImpl::write_grid(
        idg::Grid& grid)
    {
        auto nr_w_layers = grid.get_w_dim();
        auto nr_correlations = grid.get_z_dim();
        auto height = grid.get_y_dim();
        auto width = grid.get_y_dim();
        assert(nr_correlations == 4);
        assert(height == width);
        auto grid_size = height;

        std::vector<float> grid_real(nr_w_layers * nr_correlations * grid_size * grid_size * sizeof(float));
        std::vector<float> grid_imag(nr_w_layers * nr_correlations * grid_size * grid_size * sizeof(float));
        for (int w = 0; w < nr_w_layers; w++) {
            #pragma omp parallel for
            for (int y = 0; y < grid_size; y++) {
                for (int x = 0; x < grid_size; x++) {
                    for (int pol = 0; pol < nr_correlations; pol++) {
                        size_t idx = w * nr_correlations * grid_size * grid_size +
                                     pol * grid_size * grid_size +
                                     y * grid_size +
                                     x;
                        grid_real[idx] = grid(w, pol, y, x).real();
                        grid_imag[idx] = grid(w, pol, y, x).imag();
                    }
                }
            }
        }
        std::cout << "writing grid to grid_real.npy and grid_imag.npy" << std::endl;
        const long unsigned leshape [] = {(long unsigned int) nr_w_layers, 4, grid_size, grid_size};
        npy::SaveArrayAsNumpy("grid_real.npy", false, 4, leshape, grid_real);
        npy::SaveArrayAsNumpy("grid_imag.npy", false, 4, leshape, grid_imag);
    }

    void BufferSetImpl::get_image(double* image)
    {
        m_get_image_watch->Start();

        double runtime = -omp_get_wtime();
#if ENABLE_VERBOSE_TIMING
        std::cout << std::setprecision(3);
#endif

        const int nr_w_layers = m_grid->get_w_dim();
        const size_t y0 = (m_padded_size-m_size)/2;
        const size_t x0 = (m_padded_size-m_size)/2;

        // Fourier transform w layers
#if ENABLE_VERBOSE_TIMING
        std::cout << "ifft w_layers";
#endif
        int batch = nr_w_layers * 4;
        double runtime_fft = -omp_get_wtime();
        ifft2f(batch, m_padded_size, m_padded_size, m_grid->data(0,0,0,0));
        runtime_fft += omp_get_wtime();
#if ENABLE_VERBOSE_TIMING
        std::cout << ", runtime: " << runtime_fft << std::endl;
#endif

        // Stack w layers
        double runtime_stacking = -omp_get_wtime();
        for (int w = 0; w < nr_w_layers; w++) {
#if ENABLE_VERBOSE_TIMING
            std::cout << "stacking w_layer: " << w+1 << "/" << nr_w_layers << std::endl;
#endif
            #pragma omp parallel for
            for (int y = 0; y < m_size; y++) {
                for (int x = 0; x < m_size; x++) {
                    // Compute phase
                    const float w_offset = (w+0.5)*m_w_step;
                    const float l = (x-((int)m_size/2)) * m_cell_size;
                    const float m = (y-((int)m_size/2)) * m_cell_size;
                    const float n = compute_n(l, -m, m_shift);
                    // evaluate n = 1.0f - sqrt(1.0 - (l * l) - (m * m));
                    // accurately for small values of l and m
                    //const float tmp = (l * l) + (m * m);
                    //const float n = tmp > 1.0 ? 1.0 : tmp / (1.0f + sqrtf(1.0f - tmp));
                    const float phase = -2*M_PI*n*w_offset;

                    // Compute phasor
                    std::complex<float> phasor(std::cos(phase), std::sin(phase));

                    // Compute inverse spheroidal
                    float inv_taper = m_inv_taper[y] * m_inv_taper[x];

                    // Check for NaN
                    #if DEBUG_NAN_GET_IMAGE
                    if (isnan(m_grid(w, 0, y+y0, x+x0)) ||
                        isnan(m_grid(w, 1, y+y0, x+x0)) ||
                        isnan(m_grid(w, 2, y+y0, x+x0)) ||
                        isnan(m_grid(w, 3, y+y0, x+x0)))
                    {
                        std::cerr << "NaN detected during w-stacking!" << std::endl;
                        std::raise(SIGFPE);
                    }
                    #endif

                    // Apply correction
                    (*m_grid)(w, 0, y+y0, x+x0) = (*m_grid)(w, 0, y+y0, x+x0) * inv_taper * phasor;
                    (*m_grid)(w, 1, y+y0, x+x0) = (*m_grid)(w, 1, y+y0, x+x0) * inv_taper * phasor;
                    (*m_grid)(w, 2, y+y0, x+x0) = (*m_grid)(w, 2, y+y0, x+x0) * inv_taper * phasor;
                    (*m_grid)(w, 3, y+y0, x+x0) = (*m_grid)(w, 3, y+y0, x+x0) * inv_taper * phasor;

                    // Add to first w-plane
                    if (w > 0) {
                        #pragma unroll
                        for (int pol = 0; pol < 4; pol++) {
                            (*m_grid)(0, pol, y+y0, x+x0) += (*m_grid)(w, pol, y+y0, x+x0);
                        }
                    }
                } // end for x
            } // end for y
        } // end for w
        runtime_stacking += omp_get_wtime();
#if ENABLE_VERBOSE_TIMING
        std::cout << "w-stacking runtime: " << runtime_stacking << std::endl;
#endif


        // Copy grid to image
#if ENABLE_VERBOSE_TIMING
        std::cout << "set image from grid";
#endif
        double runtime_copy = -omp_get_wtime();
        #pragma omp parallel for
        for (int y = 0; y < m_size; y++) {
            for (int x = 0; x < m_size; x++) {
            // Stokes I
            image[0*m_size*m_size + m_size*y+x] = 0.5 * ((*m_grid)(0,0,y+y0,x+x0).real() + (*m_grid)(0,3,y+y0,x+x0).real());
            // Stokes Q
            image[1*m_size*m_size + m_size*y+x] = 0.5 * ((*m_grid)(0,0,y+y0,x+x0).real() - (*m_grid)(0,3,y+y0,x+x0).real());
            // Stokes U
            image[2*m_size*m_size + m_size*y+x] = 0.5 * ((*m_grid)(0,1,y+y0,x+x0).real() + (*m_grid)(0,2,y+y0,x+x0).real());
            // Stokes V
            image[3*m_size*m_size + m_size*y+x] = 0.5 * (-(*m_grid)(0,1,y+y0,x+x0).imag() + (*m_grid)(0,2,y+y0,x+x0).imag());

            // Check for NaN
            #if DEBUG_NAN_GET_IMAGE
            if (std::isnan(image[0*m_size*m_size + m_size*y+x]) ||
                std::isnan(image[1*m_size*m_size + m_size*y+x]) ||
                std::isnan(image[2*m_size*m_size + m_size*y+x]) ||
                std::isnan(image[3*m_size*m_size + m_size*y+x]))
            {
                std::cerr << "NaN detected during setting stokes!" << std::endl;
                std::raise(SIGFPE);
            }
            #endif

            } // end for x
        } // end for y
        runtime_copy += omp_get_wtime();
#if ENABLE_VERBOSE_TIMING
        std::cout << ", runtime: " << runtime_copy << std::endl;
#endif

        // Report overall runtime
        runtime += omp_get_wtime();
#if ENABLE_VERBOSE_TIMING
        std::cout << "runtime " << __func__ << ": " << runtime << std::endl;
#endif
        m_get_image_watch->Pause();
    }

    void BufferSetImpl::finished()
    {
        if (m_buffer_set_type == BufferSetType::gridding)
        {
            for (auto& buffer : m_gridderbuffers )
            {
                buffer->finished();
            }
        }
        else
        {
            for (auto& buffer : m_degridderbuffers )
            {
                buffer->finished();
            }
        }
    }

    void BufferSetImpl::init_compute_avg_beam(compute_flags flag)
    {
        m_do_compute_avg_beam = true;
        m_do_gridding = (flag != compute_flags::compute_only);
        m_average_beam = std::vector<std::complex<float>>(m_subgridsize*m_subgridsize*16, 0.0);
    }


    void BufferSetImpl::finalize_compute_avg_beam()
    {
        m_matrix_inverse_beam = std::make_shared<std::vector<std::complex<float>>>(m_average_beam);
        m_scalar_beam = std::make_shared<std::vector<float>>(m_size * m_size);
        std::vector<std::complex<float>> scalar_beam_subgrid (m_subgridsize*m_subgridsize, 1.0);
        std::vector<std::complex<float>> scalar_beam_padded (m_padded_size*m_padded_size, 0.0);

        m_do_compute_avg_beam = false;
        m_do_gridding = true;

        {
            const long unsigned leshape [] = {m_subgridsize, m_subgridsize,4,4};
            npy::SaveArrayAsNumpy("beam.npy", false, 4, leshape, *m_matrix_inverse_beam);
        }

        for (int i = 0; i < m_subgridsize * m_subgridsize; i++)
        {

//             LAPACKE_cgetrf( LAPACK_COL_MAJOR, 4, 4, (lapack_complex_float*) data, 4, ipiv);
//             extern void cgetrf( int* m, int* n, std::complex<float>* a,
//                     int* lda, int* ipiv, int *info );


            std::complex<float> *data = m_matrix_inverse_beam->data() + i*16;
            int n = 4;
            int info;
            int ipiv[4];
            cgetrf_( &n, &n, data, &n, ipiv, &info );


//             LAPACKE_cgetri( LAPACK_COL_MAJOR, 4, (lapack_complex_float*) data, 4, ipiv);
//             extern void cgetri( int* n, std::complex<float>* a, int* lda,
//                                 const int* ipiv, std::complex<float>* work,
//                                 int* lwork, int *info );

            int lwork = -1;
            std::complex<float> wkopt;
            cgetri_(&n, data, &n, ipiv, &wkopt, &lwork, &info );
            lwork = int(wkopt.real());
            std::vector<std::complex<float>> work(lwork);
            cgetri_(&n, data, &n, ipiv, work.data(), &lwork, &info );

            // NOTE: there is a sign flip between the idg subgrids and the master image
            scalar_beam_subgrid[m_subgridsize*m_subgridsize-i-1] = 1.0/sqrt(data[0].real() + data[3].real() + data[12].real() + data[15].real());

            #pragma omp simd
            for(size_t j=0; j<16; j++)
            {
                data[j] *= scalar_beam_subgrid[m_subgridsize*m_subgridsize-i-1];
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
        for(int i = 0; i < int(m_subgridsize); i++)
        {
            for(int j = 0; j < int(m_subgridsize); j++)
            {
                scalar_beam_subgrid[i*m_subgridsize + j] *= m_taper_subgrid[i] * m_taper_subgrid[j];
            }
        }

        // 2. fft
        fft2f(m_subgridsize, scalar_beam_subgrid.data());

        // 3. multiply by phase gradient for half pixel shift
        for(size_t i=0; i<m_subgridsize; i++)
        {
            for(size_t j=0; j<m_subgridsize; j++)
            {
                float phase = -M_PI*((float(i)+float(j))/m_subgridsize-1.0);

                // Compute phasor
                std::complex<float> phasor(std::cos(phase), std::sin(phase));

                scalar_beam_subgrid[i*m_subgridsize+j] *= phasor/float(m_subgridsize*m_subgridsize);

            }
        }

        // 4. zero pad
        {
            size_t offset = (m_padded_size - m_subgridsize)/2;
            for(size_t i=0; i<m_subgridsize; i++)
            {
                for(size_t j=0; j<m_subgridsize; j++)
                {
                    scalar_beam_padded[(i+offset)*m_padded_size+(j+offset)] = scalar_beam_subgrid[i*m_subgridsize+j];
                }
            }
        }

        // 5. ifft
        ifft2f(m_padded_size, scalar_beam_padded.data());

        // 6. divide out taper and normalize
        {
            size_t offset = (m_padded_size - m_size)/2;
            float x_center = m_size/2;
            float y_center = m_size/2;
            float center_value = scalar_beam_padded[(y_center+offset)*m_padded_size + x_center + offset].real() * m_inv_taper[x_center] * m_inv_taper[y_center];
            float normalization = 1.0/center_value;
            for (size_t y = 0; y < m_size; y++)
            {
                for (size_t x = 0; x < m_size; x++)
                {
                    (*m_scalar_beam)[m_size*y+x] = scalar_beam_padded[(y+offset)*m_padded_size + x + offset].real() * m_inv_taper[x] * m_inv_taper[y] * normalization;
                }
            }

            // normalize matrix beam as well
            for (int i = 0; i < m_subgridsize * m_subgridsize; i++)
            {
                std::complex<float> *data = m_matrix_inverse_beam->data() + i*16;
                #pragma omp simd
                for(size_t j=0; j<16; j++)
                {
                    data[j] *= normalization;
                }
            }
        }

        {
            const long unsigned leshape [] = {m_size, m_size};
            npy::SaveArrayAsNumpy("scalar_beam.npy", false, 2, leshape, *m_scalar_beam);
        }

        m_avg_aterm_correction = Array4D<std::complex<float>>( m_matrix_inverse_beam->data(), m_subgridsize, m_subgridsize, 4, 4);
        m_proxy->set_avg_aterm_correction(m_avg_aterm_correction);

        {
            const long unsigned leshape [] = {m_subgridsize, m_subgridsize,4,4};
            npy::SaveArrayAsNumpy("beam_inv.npy", false, 4, leshape, *m_matrix_inverse_beam);
        }

    }

    void BufferSetImpl::set_matrix_inverse_beam(std::shared_ptr<std::vector<std::complex<float>>> matrix_inverse_beam)
    {
        m_matrix_inverse_beam = matrix_inverse_beam;
        m_avg_aterm_correction = Array4D<std::complex<float>>( m_matrix_inverse_beam->data(), m_subgridsize, m_subgridsize, 4, 4);
        m_proxy->set_avg_aterm_correction(m_avg_aterm_correction);
    }

    void BufferSetImpl::unset_matrix_inverse_beam()
    {
        m_matrix_inverse_beam.reset();
        m_avg_aterm_correction = Array4D<std::complex<float>>(0,0,0,0);
        m_proxy->unset_avg_aterm_correction();
    }

    void BufferSetImpl::report_runtime() {
        std::clog << "avg beam:   " << m_avg_beam_watch->ToString() << std::endl;
        std::clog << "plan:       " << m_plan_watch->ToString() << std::endl;
        std::clog << "gridding:   " << m_gridding_watch->ToString() << std::endl;
        std::clog << "degridding: " << m_degridding_watch->ToString() << std::endl;
        std::clog << "set image:  " << m_get_image_watch->ToString() << std::endl;
        std::clog << "get image:  " << m_set_image_watch->ToString() << std::endl;
    }


} // namespace api
} // namespace idg
