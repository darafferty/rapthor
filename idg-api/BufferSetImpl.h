#ifndef IDG_API_BUFFERSETIMPL_H_
#define IDG_API_BUFFERSETIMPL_H_

#include <vector>
#include <memory>

#include "idg-common.h"


#include "BufferSet.h"

namespace idg {
namespace api {

    class BufferSetImpl : public virtual BufferSet {
    public:
        BufferSetImpl(
            Type architecture, 
            size_t bufferTimesteps, 
            std::vector<std::vector<double>> bands,
            int nr_stations,
            size_t width, 
            float cellsize, 
            float max_baseline,
            float max_w,
            options_type &options,
            BufferSetType buffer_set_type);

        DegridderBuffer* get_degridder(int i);
        GridderBuffer* get_gridder(int i);

        virtual void set_image(const double* image);
        virtual void get_image(double* image);
        virtual void finished();

    private:
        BufferSetType m_buffer_set_type;
        std::vector<std::unique_ptr<GridderBuffer>> m_gridderbuffers;
        std::vector<std::unique_ptr<DegridderBuffer>> m_degridderbuffers;
        std::vector<float> m_taper_subgrid;
        std::vector<float> m_taper_grid;
        Grid m_grid;
        float m_image_size;
        float m_cell_size;
        float m_w_step;
        size_t m_width;
        size_t m_padded_width;
    };

} // namespace api
} // namespace idg

#endif
