#ifndef IDG_API_BUFFERSET_H_
#define IDG_API_BUFFERSET_H_


#include "GridderBuffer.h"
#include "DegridderBuffer.h"

namespace idg {
namespace api {

    enum class BufferSetType
    {
        gridding,
        degridding
    };
    
    
    class BufferSet {
    public:
        BufferSet(
            Type architecture, 
            size_t bufferTimesteps, 
            std::vector<std::vector<double>> bands,
            int nr_stations,
            size_t width, 
            float cellsize, 
            float max_w,         
            BufferSetType buffer_set_type);
        
        DegridderBuffer* get_degridder(int i);
        GridderBuffer* get_gridder(int i);
    
        virtual void set_image(const double* image);
        virtual void get_image(double* image);
        virtual void finished();
        
    private:
        BufferSetType m_buffer_set_type;
        std::vector<std::unique_ptr<Buffer>> m_buffers;
        std::vector<float> m_taper_subgrid;
        std::vector<float> m_taper_grid;
        Grid m_grid;
        float m_image_size;
        float m_w_step;
    };
    
    

} // namespace api
} // namespace idg

#endif
