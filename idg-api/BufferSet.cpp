#include "BufferSet.h"

#include <cmath>

#include "taper.h"

namespace idg {
namespace api {

    BufferSet::BufferSet(
        Type architecture, 
        size_t bufferTimesteps, 
        std::vector<std::vector<double>> bands,
        int nr_stations,
        size_t width, 
        float cell_size, 
        float max_w,         
        BufferSetType buffer_set_type) :
        m_grid(0,0,0,0,0),
        m_buffer_set_type(buffer_set_type)
    {
        // 
        m_image_size = cell_size * width;

        // this cuts the w kernel approximately at the 1% level        
        const float max_w_size = 2*max_w * m_image_size * m_image_size;

        // some heuristic to set kernel size
        // square root splits the w_kernel evenly over wstack and wprojection
        // still needs a bit more thinking, and better motivation.
        // but for now does something reasonable
        float w_kernel_size = std::max(8, int(std::round(2*std::sqrt(max_w_size))));

        m_w_step = 2 * w_kernel_size / (m_image_size * m_image_size);
        int nr_w_layers = std::ceil(max_w / m_w_step);
        m_w_step = max_w / nr_w_layers;
        w_kernel_size = 0.5*m_w_step * m_image_size * m_image_size;

        int max_nr_channels = 0;
        for (auto band : bands )
        {
            max_nr_channels = std::max(max_nr_channels, int(band.size()));
        }

        float taper_kernel_size = 7.0;
        float padding = 1.25;

        float a_term_kernel_size = 0.0;

        // idg library's Plan does not account for channels, when mapping visibilities onto subgrids
        // therefore the maximum number of channels is added to the kernel size
        float kernel_size = taper_kernel_size + w_kernel_size + a_term_kernel_size + max_nr_channels + 8.0;

        // reserved space in subgrid for time 
        float uv_span_time = 8.0;


        int subgridsize = int(std::ceil((kernel_size + uv_span_time)/8.0))*8;

        m_grid = Grid(nr_w_layers,4,width,width);

        m_taper_subgrid.resize(subgridsize);
        m_taper_grid.resize(width);
        init_optimal_taper_1D(subgridsize, width, taper_kernel_size, padding, m_taper_subgrid.data(), m_taper_grid.data());

        std::vector<float> taper;
        taper.resize(subgridsize * subgridsize);
        for(int i = 0; i < int(subgridsize); i++)
            for(int j = 0; j < int(subgridsize); j++)
                taper[i*subgridsize + j] = m_taper_subgrid[i] * m_taper_subgrid[j];

        m_buffers.reserve(bands.size());

        for (auto band : bands )
        {
            Buffer *buffer;
            if (buffer_set_type == BufferSetType::gridding)
            {
                buffer = new GridderBuffer(architecture, bufferTimesteps);
            }
            else
            {
                buffer = new DegridderBuffer(architecture, bufferTimesteps);
            }

            m_buffers.push_back(std::unique_ptr<Buffer>(buffer));

            buffer->set_subgrid_size(subgridsize);

            buffer->set_frequencies(band);
            buffer->set_stations(nr_stations);
            buffer->set_cell_size(cell_size, cell_size);
            buffer->set_w_step(m_w_step);
            buffer->set_kernel_size(kernel_size);
            buffer->set_spheroidal(subgridsize, subgridsize, taper.data());
            buffer->set_grid(&m_grid);
            buffer->bake();
        }
    }

    GridderBuffer* BufferSet::get_gridder(int i)
    {
        if (m_buffer_set_type != BufferSetType::gridding)
        {
            throw(std::logic_error("BufferSet is not of gridding type"));
        }
        return static_cast<GridderBuffer*>(m_buffers[i].get());
    }

    DegridderBuffer* BufferSet::get_degridder(int i)
    {
        if (m_buffer_set_type != BufferSetType::degridding)
        {
            throw(std::logic_error("BufferSet is not of degridding type"));
        }
        return static_cast<DegridderBuffer*>(m_buffers[i].get());
    }

    void BufferSet::set_image(const double* image) 
    {
        //Convert from stokes to linear into w plane 0

        m_grid.init(0.0);

        int width = m_grid.get_x_dim();
        int nr_w_layers = m_grid.get_nr_w_layers();

        // Stokes I
        for(int i=0; i < width; i++)
        {
            for(int j=0; j < width; j++)
            {
                m_grid(0,0,i,j) = image[width*i+j];
                m_grid(0,3,i,j) = image[width*i+j];
            }
        }

        // Stokes Q
        for(int i=0; i < width; i++)
        {
            for(int j=0; j < width; j++)
            {
                m_grid(0,0,i,j) += image[width*width + width*i+j];
                m_grid(0,3,i,j) -= image[width*width + width*i+j];
            }
        }

        // Stokes U
        for(int i=0; i < width; i++)
        {
            for(int j=0; j < width; j++)
            {
                m_grid(0,1,i,j) = image[2*width*width + width*i+j];
                m_grid(0,2,i,j) = image[2*width*width + width*i+j];
            }
        }

        // Stokes V
        for(int i=0; i < width; i++)
        {
            for(int j=0; j < width; j++)
            {
                m_grid(0,1,i,j).imag(image[3*width*width + width*i+j]);
                m_grid(0,2,i,j).imag(-image[3*width*width + width*i+j]);
            }
        }

        //Copy to other w planes and multiply by w term

        std::vector<float> inv_spheroidal(width, 0.0);

        for(size_t i=0; i < width ; ++i)
        {
            float y = m_taper_grid[i];
            if (y > 1e-3) inv_spheroidal[i] = 1.0/y;
        }

        for(int w_layer=nr_w_layers - 1; w_layer >= 0; w_layer--)
        {
            std::cout << "w_layer: " << w_layer << std::endl;
            const float w_offset = (w_layer+0.5)*m_w_step;
            for(int i=0; i < width; i++)
            {
                for(int j=0; j < width; j++)
                {
                    const float l = (i-(width/2)) * m_image_size/width;
                    const float m = (j-(width/2)) * m_image_size/width;
                    // evaluate n = 1.0f - sqrt(1.0 - (l * l) - (m * m));
                    // accurately for small values of l and m
                    const float tmp = (l * l) + (m * m);
                    const float n = tmp / (1.0f + sqrtf(1.0f - tmp));

                    float phase = 2*M_PI*n*w_offset;
                    std::complex<float> phasor(std::cos(phase), std::sin(phase));
                    for(int pol=0; pol<4; pol++)
                    {
                        m_grid(w_layer, pol, i, j) = m_grid(0, pol, i, j) * inv_spheroidal[i] * inv_spheroidal[j] * phasor;
                    }
                }
            }
            m_buffers[0]->fft_grid(4, width, width, &m_grid(w_layer,0,0,0));
        }
    };

    void BufferSet::get_image(double* image) 
    {
        int width = m_grid.get_x_dim();
        int nr_w_layers = m_grid.get_nr_w_layers();

        std::vector<float> inv_spheroidal(width, 0.0);

        for(size_t i=0; i < width ; ++i)
        {
            float y = m_taper_grid[i];
            if (y > 1e-3) inv_spheroidal[i] = 1.0/y;
        }

        for(int w_layer=0; w_layer < nr_w_layers; w_layer++)
        {
            m_buffers[0]->ifft_grid(4, width, width, &m_grid(w_layer,0,0,0));
        }
        for(int w_layer=0; w_layer < nr_w_layers; w_layer++)
        {
            std::cout << "w_layer: " << w_layer << std::endl;
            const float w_offset = (w_layer+0.5)*m_w_step;
            #pragma omp parallel for
            for(int i=0; i < width; i++)
            {
                for(int j=0; j < width; j++)
                {
                    const float l = (i-(width/2)) * m_image_size/width;
                    const float m = (j-(width/2)) * m_image_size/width;
                    // evaluate n = 1.0f - sqrt(1.0 - (l * l) - (m * m));
                    // accurately for small values of l and m
                    const float tmp = (l * l) + (m * m);
                    const float n = tmp / (1.0f + sqrtf(1.0f - tmp));

                    float phase = -2*M_PI*n*w_offset;
                    std::complex<float> phasor(std::cos(phase), std::sin(phase));
                    for(int pol=0; pol<4; pol++)
                    {
                         m_grid(w_layer, pol, i, j) = m_grid(w_layer, pol, i, j) * inv_spheroidal[i] * inv_spheroidal[j] * phasor;
                    }
                }
            }
        }

        for(int w_layer=1; w_layer < nr_w_layers; w_layer++)
        {
            for(int i=0; i < width; i++)
            {
                for(int j=0; j < width; j++)
                {
                    for(int pol=0; pol<4; pol++)
                    {
                        m_grid(0, pol, i, j) += m_grid(w_layer, pol, i, j);
                    }
                }
            }
        }

        // Stokes I
        for(int i=0; i < width; i++)
        {
            for(int j=0; j < width; j++)
            {
                image[width*i+j] = 0.5 * (m_grid(0,0,i,j).real() + m_grid(0,3,i,j).real());
            }
        }

        // Stokes Q
        for(int i=0; i < width; i++)
        {
            for(int j=0; j < width; j++)
            {
                image[width*width + width*i+j] = 0.5 * (m_grid(0,0,i,j).real() - m_grid(0,3,i,j).real());
            }
        }

        // Stokes U
        for(int i=0; i < width; i++)
        {
            for(int j=0; j < width; j++)
            {
                image[2*width*width + width*i+j] = 0.5 * (m_grid(0,1,i,j).real() + m_grid(0,2,i,j).real());
            }
        }

        // Stokes V
        for(int i=0; i < width; i++)
        {
            for(int j=0; j < width; j++)
            {
                image[3*width*width + width*i+j] = 0.5 * (m_grid(0,1,i,j).imag() - m_grid(0,2,i,j).imag());;
            }
        }
    }

    void BufferSet::finished()
    {
        for (auto& buffer : m_buffers )
        {
            buffer->finished();
        }
    }

} // namespace api
} // namespace idg
