#include "BufferSetImpl.h"
#include "GridderBufferImpl.h"
#include "DegridderBufferImpl.h"

#include <complex>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include "taper.h"

namespace idg {
namespace api {

    BufferSet* BufferSet::create(Type architecture)
    {
        return new BufferSetImpl(architecture);
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
        m_grid(0,0,0,0,0),
        m_proxy(nullptr)
    {}

    proxy::Proxy* BufferSetImpl::get_proxy(int subgridsize)
    {
        if (m_proxies.count(subgridsize) == 0)
        {
            m_proxies[subgridsize] = create_proxy(subgridsize);
        }
        return m_proxies[subgridsize];
    }

    proxy::Proxy* BufferSetImpl::create_proxy(int subgridsize)
    {
        proxy::Proxy* proxy;
        int nr_correlations = 4;
        CompileConstants constants(nr_correlations, subgridsize);

        if (m_architecture == Type::CPU_REFERENCE) {
            #if defined(BUILD_LIB_CPU)
                proxy = new proxy::cpu::Reference(constants);
            #else
                throw std::runtime_error("Can not create CPU_REFERENCE proxy. idg-lib was built with BUILD_LIB_CPU=OFF");
            #endif
        } else if (m_architecture == Type::CPU_OPTIMIZED) {
            #if defined(BUILD_LIB_CPU)
                proxy = new proxy::cpu::Optimized(constants);
            #else
                throw std::runtime_error("Can not create CPU_OPTIMIZED proxy. idg-lib was built with BUILD_LIB_CPU=OFF");
            #endif
        }
        if (m_architecture == Type::CUDA_GENERIC) {
            #if defined(BUILD_LIB_CUDA)
                proxy = new proxy::cuda::Generic(constants);
            #else
                throw std::runtime_error("Can not create CUDA_GENERIC proxy. idg-lib was built with BUILD_LIB_CUDA=OFF");
            #endif
        }
        if (m_architecture == Type::HYBRID_CUDA_CPU_OPTIMIZED) {
            #if defined(BUILD_LIB_CPU) && defined(BUILD_LIB_CUDA)
                // cpu proxy will be deleted by hybrid proxy destructor
                proxy::cpu::CPU *cpu_proxy = new proxy::cpu::Optimized(constants);
                proxy = new proxy::hybrid::HybridCUDA(cpu_proxy, constants);
                delete proxy;
                proxy = new proxy::cpu::Optimized(constants);
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
                proxy = new proxy::opencl::Generic(constants);
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
        options_type &options)
    {
        const float taper_kernel_size = 7.0;
        const float a_term_kernel_size = 0.0;

        m_size = size;

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

        std::cout << "m_padded_size: " << m_padded_size << std::endl;
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

        std::cout << "nr_w_layers: " << nr_w_layers << std::endl;

        m_w_step = max_w / nr_w_layers;
        w_kernel_size = 0.5*m_w_step * m_image_size * m_image_size;

        // DEBUG no w-stacking
//         w_kernel_size = max_w_size;
//         nr_w_layers = 1;
//         m_w_step = 0.0;

        m_kernel_size = taper_kernel_size + w_kernel_size + a_term_kernel_size;

        // reserved space in subgrid for time 
        m_uv_span_time = 8.0;

        m_uv_span_frequency = 24.0;

        m_subgridsize = int(std::ceil((m_kernel_size + m_uv_span_time + m_uv_span_frequency)/8.0))*8;

        m_proxy = get_proxy(m_subgridsize);

        m_grid = Grid(nr_w_layers,4,m_padded_size,m_padded_size);

        m_taper_subgrid.resize(m_subgridsize);
        m_taper_grid.resize(m_padded_size);
        init_optimal_taper_1D(m_subgridsize, m_padded_size, m_size, taper_kernel_size, m_taper_subgrid.data(), m_taper_grid.data());
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
                GridderBufferImpl *gridderbuffer = new GridderBufferImpl(m_proxy, bufferTimesteps);
                m_gridderbuffers.push_back(std::unique_ptr<GridderBuffer>(gridderbuffer));
                buffer = gridderbuffer;
            }
            else
            {
                DegridderBufferImpl *degridderbuffer = new DegridderBufferImpl(m_proxy, bufferTimesteps);
                m_degridderbuffers.push_back(std::unique_ptr<DegridderBuffer>(degridderbuffer));
                buffer = degridderbuffer;
            }


            buffer->set_subgrid_size(m_subgridsize);

            buffer->set_frequencies(band);
            buffer->set_stations(nr_stations);
            buffer->set_cell_size(m_cell_size, m_cell_size);
            buffer->set_w_step(m_w_step);
            buffer->set_kernel_size(m_kernel_size);
            buffer->set_spheroidal(m_subgridsize, m_subgridsize, taper.data());
            buffer->set_grid(&m_grid);
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

    void BufferSetImpl::set_image(const double* image) 
    {
        double runtime = -omp_get_wtime();

        //Convert from stokes to linear into w plane 0

        m_grid.init(0.0);

        const int nr_w_layers = m_grid.get_w_dim();
        const size_t i0 = (m_padded_size-m_size)/2;
        const size_t j0 = (m_padded_size-m_size)/2;

        // Stokes I
        #pragma omp parallel for
        for(size_t i=0; i < m_size; i++)
        {
            #pragma omp simd
            for(size_t j=0; j < m_size; j++)
            {
                m_grid(0,0,i+i0,j+j0) = image[m_size*i+j];
                m_grid(0,3,i+i0,j+j0) = image[m_size*i+j];
            }
        }

        // Stokes Q
        #pragma omp parallel for
        for(size_t i=0; i < m_size; i++)
        {
            #pragma omp simd
            for(size_t j=0; j < m_size; j++)
            {
                m_grid(0,0,i+i0,j+j0) += image[m_size*m_size + m_size*i+j];
                m_grid(0,3,i+i0,j+j0) -= image[m_size*m_size + m_size*i+j];
            }
        }

        // Stokes U
        #pragma omp parallel for
        for(size_t i=0; i < m_size; i++)
        {
            #pragma omp simd
            for(size_t j=0; j < m_size; j++)
            {
                m_grid(0,1,i+i0,j+j0) = image[2*m_size*m_size + m_size*i+j];
                m_grid(0,2,i+i0,j+j0) = image[2*m_size*m_size + m_size*i+j];
            }
        }

        // Stokes V
        #pragma omp parallel for
        for(size_t i=0; i < m_size; i++)
        {
            #pragma omp simd
            for(size_t j=0; j < m_size; j++)
            {
                m_grid(0,1,i+i0,j+j0).imag(image[3*m_size*m_size + m_size*i+j]);
                m_grid(0,2,i+i0,j+j0).imag(-image[3*m_size*m_size + m_size*i+j]);
            }
        }

        //Copy to other w planes and multiply by w term

        std::vector<float> inv_spheroidal(m_size, 0.0);

        for(size_t i=0; i < m_size ; ++i)
        {
            float y = m_taper_grid[i+i0];
            inv_spheroidal[i] = 1.0/y;
        }

        for(int w_layer=nr_w_layers - 1; w_layer >= 0; w_layer--)
        {
            std::cout << "w_layer: " << w_layer << std::endl;
            const float w_offset = (w_layer+0.5)*m_w_step;
            #pragma omp parallel for
            for(int i=0; i < m_size; i++)
            {
                #pragma omp simd
                for(int j=0; j < m_size; j++)
                {
                    const float l = (i-((int)m_size/2)) * m_cell_size;
                    const float m = (j-((int)m_size/2)) * m_cell_size;
                    // evaluate n = 1.0f - sqrt(1.0 - (l * l) - (m * m));
                    // accurately for small values of l and m
                    const float tmp = (l * l) + (m * m);
                    const float n = tmp / (1.0f + sqrtf(1.0f - tmp));

                    float phase = 2*M_PI*n*w_offset;
                    std::complex<float> phasor(std::cos(phase), std::sin(phase));
                    float inv_spheroidal2 = inv_spheroidal[i] * inv_spheroidal[j];

//                     for(int pol=0; pol<4; pol++)
//                     {
//                         m_grid(w_layer, pol, i+i0, j+j0) = m_grid(0, pol, i+i0, j+j0) * inv_spheroidal2 * phasor;
//                     }
                    m_grid(w_layer, 0, i+i0, j+j0) = m_grid(0, 0, i+i0, j+j0) * inv_spheroidal2 * phasor;
                    m_grid(w_layer, 1, i+i0, j+j0) = m_grid(0, 1, i+i0, j+j0) * inv_spheroidal2 * phasor;
                    m_grid(w_layer, 2, i+i0, j+j0) = m_grid(0, 2, i+i0, j+j0) * inv_spheroidal2 * phasor;
                    m_grid(w_layer, 3, i+i0, j+j0) = m_grid(0, 3, i+i0, j+j0) * inv_spheroidal2 * phasor;

                }
            }
            m_degridderbuffers[0]->fft_grid(4, m_padded_size, m_padded_size, &m_grid(w_layer,0,0,0));
        }

        runtime += omp_get_wtime();
        std::cout << "runtime " << __func__ << ": " << std::setprecision(3) << runtime << std::endl;
    };

    void BufferSetImpl::get_image(double* image) 
    {
        double runtime = -omp_get_wtime();
        
        int nr_w_layers = m_grid.get_w_dim();

        const size_t i0 = (m_padded_size-m_size)/2;
        const size_t j0 = (m_padded_size-m_size)/2;

        std::vector<float> inv_spheroidal(m_size, 0.0);

        for(size_t i=0; i < m_size ; ++i)
        {
            float y = m_taper_grid[i+(m_padded_size-m_size)/2];
            inv_spheroidal[i] = 1.0/y;
        }

        for(int w_layer=0; w_layer < nr_w_layers; w_layer++)
        {
            std::cout << "w_layer: " << w_layer << "/" << nr_w_layers << std::endl;
            m_gridderbuffers[0]->ifft_grid(4, m_padded_size, m_padded_size, &m_grid(w_layer,0,0,0));

            const float w_offset = (w_layer+0.5)*m_w_step;
            #pragma omp parallel for
            for(int i=0; i < m_size; i++)
            {
                #pragma omp simd
                for(int j=0; j < m_size; j++)
                {
                    const float l = (i-((int)m_size/2)) * m_cell_size;
                    const float m = (j-((int)m_size/2)) * m_cell_size;
                    // evaluate n = 1.0f - sqrt(1.0 - (l * l) - (m * m));
                    // accurately for small values of l and m
                    const float tmp = (l * l) + (m * m);
                    const float n = tmp / (1.0f + sqrtf(1.0f - tmp));

                    float phase = -2*M_PI*n*w_offset;
                    std::complex<float> phasor(std::cos(phase), std::sin(phase));

                    float inv_spheroidal2 = inv_spheroidal[i] * inv_spheroidal[j];

                    m_grid(w_layer, 0, i+i0, j+j0) = m_grid(w_layer, 0, i+i0, j+j0) * inv_spheroidal2 * phasor;
                    m_grid(w_layer, 1, i+i0, j+j0) = m_grid(w_layer, 1, i+i0, j+j0) * inv_spheroidal2 * phasor;
                    m_grid(w_layer, 2, i+i0, j+j0) = m_grid(w_layer, 2, i+i0, j+j0) * inv_spheroidal2 * phasor;
                    m_grid(w_layer, 3, i+i0, j+j0) = m_grid(w_layer, 3, i+i0, j+j0) * inv_spheroidal2 * phasor;
                }
            }

            if (w_layer>0)
            {
                #pragma omp parallel for
                for(size_t i=0; i < m_size; i++)
                {
                    #pragma omp simd
                    for(size_t j=0; j < m_size; j++)
                    {
                        for(int pol=0; pol<4; pol++)
                        {
                            m_grid(0, pol, i+i0, j+j0) += m_grid(w_layer, pol, i+i0, j+j0);
                        }
                    }
                }
            }
        }

        // Stokes I
        #pragma omp parallel for
        for(size_t i=0; i < m_size; i++)
        {
            #pragma omp simd
            for(size_t j=0; j < m_size; j++)
            {
                image[m_size*i+j] = 0.5 * (m_grid(0,0,i+i0,j+j0).real() + m_grid(0,3,i+i0,j+j0).real());
            }
        }

        // Stokes Q
        #pragma omp parallel for
        for(size_t i=0; i < m_size; i++)
        {
            #pragma omp simd
            for(size_t j=0; j < m_size; j++)
            {
                image[m_size*m_size + m_size*i+j] = 0.5 * (m_grid(0,0,i+i0,j+j0).real() - m_grid(0,3,i+i0,j+j0).real());
            }
        }

        // Stokes U
        #pragma omp parallel for
        for(size_t i=0; i < m_size; i++)
        {
            #pragma omp simd
            for(size_t j=0; j < m_size; j++)
            {
                image[2*m_size*m_size + m_size*i+j] = 0.5 * (m_grid(0,1,i+i0,j+j0).real() + m_grid(0,2,i+i0,j+j0).real());
            }
        }

        // Stokes V
        #pragma omp parallel for
        for(size_t i=0; i < m_size; i++)
        {
            #pragma omp simd
            for(size_t j=0; j < m_size; j++)
            {
                image[3*m_size*m_size + m_size*i+j] = 0.5 * (m_grid(0,1,i+i0,j+j0).imag() - m_grid(0,2,i+i0,j+j0).imag());
            }
        }
        runtime += omp_get_wtime();
        std::cout << "runtime " << __func__ << ": " << runtime << std::endl;
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

} // namespace api
} // namespace idg
