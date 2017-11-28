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
        BufferSetImpl(Type architecture);

        virtual ~BufferSetImpl() {};

        DegridderBuffer* get_degridder(int i);
        GridderBuffer* get_gridder(int i);

        virtual void init(
            size_t width,
            float cellsize,
            float max_w,
            options_type &options);

        virtual void init_buffers(
            size_t bufferTimesteps,
            std::vector<std::vector<double>> bands,
            int nr_stations,
            float max_baseline,
            options_type &options,
            BufferSetType buffer_set_type);

        virtual void set_image(const double* image);
        virtual void get_image(double* image);
        virtual void finished();

        virtual int get_subgridsize() { return m_subgridsize; }
        virtual float get_subgrid_pixelsize() { return m_image_size/m_subgridsize; }

    private:

        virtual void fft_grid(
            size_t nr_polarizations    = 0,
            size_t height              = 0,
            size_t width               = 0,
            std::complex<float> *grid = nullptr);

        virtual void ifft_grid(
            size_t nr_polarizations    = 0,
            size_t height              = 0,
            size_t width               = 0,
            std::complex<float> *grid = nullptr);


        proxy::Proxy* get_proxy(int subgridsize);
        proxy::Proxy* create_proxy(int subgridsize);

        Type m_architecture;
        proxy::Proxy* m_proxy;
        std::map<int, proxy::Proxy*> m_proxies;
        BufferSetType m_buffer_set_type;
        std::vector<std::unique_ptr<GridderBuffer>> m_gridderbuffers;
        std::vector<std::unique_ptr<DegridderBuffer>> m_degridderbuffers;
        std::vector<float> m_taper_subgrid;
        std::vector<float> m_taper_grid;
        Grid m_grid;
        int m_subgridsize;
        float m_image_size;
        float m_cell_size;
        float m_w_step;
        size_t m_size;
        size_t m_padded_size;
        float m_kernel_size;
        float m_uv_span_time;
        float m_uv_span_frequency;
    };

} // namespace api
} // namespace idg

#endif
