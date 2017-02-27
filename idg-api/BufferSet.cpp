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
        m_grid(0,0,0,0,0)
    {
        // TODO determine number of w layers and subgrid size
        
        // 
        m_image_size = cell_size * width;

        // this cuts the w kernel approximately at the 1% level        
        const float max_w_size = max_w * m_image_size * m_image_size;
        
        // some heuristic to set kernel size
        // square root splits the w_kernel evenly over wstack and wprojection
        // still needs a bit more thinking, and better motivation.
        // but for now does something reasonable
        float w_kernel_size = std::max(8, int(std::round(std::sqrt(max_w_size))));
        
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
        float kernel_size = taper_kernel_size + w_kernel_size + a_term_kernel_size + max_nr_channels;
        
        // reserved space in subgrid for time 
        float uv_span_time = 8.0;
        
        
        int subgridsize = int(std::ceil((kernel_size + uv_span_time)/8.0))*8;
        
        m_grid = Grid(nr_w_layers,4,width,width);
        
        m_taper_subgrid.resize(subgridsize);
        m_taper_grid.resize(width);
        init_optimal_taper_1D(subgridsize, width, taper_kernel_size, padding, m_taper_subgrid.data(), m_taper_grid.data());
       
//        init_optimal_gridding_taper_1D(_subgridSize, width, 7.0, _taper_subgrid.data(), _taper_grid.data());
    
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
                buffer = new GridderBuffer(Type::HYBRID_CUDA_CPU_OPTIMIZED, 512);
            }
            else
            {
                buffer = new DegridderBuffer(Type::HYBRID_CUDA_CPU_OPTIMIZED, 512);
            }
            
            m_buffers.push_back(std::unique_ptr<Buffer>(buffer));

            buffer->set_subgrid_size(subgridsize);
            buffer->bake();

            buffer->set_frequencies(band);
            buffer->set_stations(nr_stations);
            buffer->set_spheroidal(subgridsize, subgridsize, taper.data());
            buffer->set_cell_size(cell_size, cell_size);
            buffer->set_kernel_size(kernel_size);
            buffer->set_grid(&m_grid);
        }
    }

    
    Buffer* BufferSet::operator[](int i)
    {
        return m_buffers[i].get();
    }
    
    void BufferSet::set_image(const double* image) 
    {
        //Convert from stokes to linear into w plane 0
        
        m_grid.init(0.0);
       
        int width = m_grid.get_x_dim();
        int nr_w_layers = m_grid.get_x_dim();

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
                m_grid(0,0,i,j) += image[width*i+j];
                m_grid(0,3,i,j) -= image[width*i+j];
            }
        }
        
        // Stokes U
        for(int i=0; i < width; i++)
        {
            for(int j=0; j < width; j++)
            {
                m_grid(0,1,i,j) = image[width*i+j];
                m_grid(0,2,i,j) = image[width*i+j];
            }
        }
        
        // Stokes V
        for(int i=0; i < width; i++)
        {
            for(int j=0; j < width; j++)
            {
                m_grid(0,1,i,j).imag(image[width*i+j]);
                m_grid(0,2,i,j).imag(-image[width*i+j]);
            }
        }
        
        //Copy to other w planes and multiply by w term
        //TODO scale with spheroidal
         
        for(int w_layer=nr_w_layers - 1; w_layer >= 0; w_layer--)
        {
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
                    std::complex<float> phasor(std::sin(phase), std::cos(phase));
                    for(int pol=0; pol<4; pol++)
                    {
                        m_grid(w_layer, pol, i, j) = m_grid(0, pol, i, j) * phasor;
                    }
                }
            }
        }
        //TODO transform
    };

    
   
    
//         double inv_spheroidal[width];
//         for(size_t i=0; i < width ; ++i)
//         {
//             double y = _taper_grid[i];
//             inv_spheroidal[i] = 0.0;
//             if (y > 1e-4) inv_spheroidal[i] = 1.0/y;
//         }
//         
//         for(size_t ii=0; ii != width * height; ++ii)
//         {
//             size_t i = ii % width;
//             size_t j = ii / width;
//             _grid[ii].real(image[ii]*inv_spheroidal[i]*inv_spheroidal[j]);
//             _grid[ii + 3*width*height].real(image[ii]*inv_spheroidal[i]*inv_spheroidal[j]);
//         }





    
} // namespace api
} // namespace idg
