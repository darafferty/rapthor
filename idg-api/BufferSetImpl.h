#ifndef IDG_API_BUFFERSETIMPL_H_
#define IDG_API_BUFFERSETIMPL_H_

#include <vector>
#include <memory>

#include "idg-common.h"


#include "BufferSet.h"

namespace idg {
namespace api {

    class BufferImpl;
    class GridderBufferImpl;
    class DegridderBufferImpl;

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

        virtual size_t get_subgridsize() { return m_subgridsize; }
        virtual float get_subgrid_pixelsize() { return m_image_size/m_subgridsize; }
        virtual void set_apply_aterm(bool do_apply) { m_apply_aterm = do_apply; }
        virtual void init_compute_avg_beam(compute_flags flag);
        virtual void finalize_compute_avg_beam();
        virtual std::shared_ptr<std::vector<float>> get_scalar_beam() const {return m_scalar_beam;}
        virtual std::shared_ptr<std::vector<std::complex<float>>> get_matrix_beam() const {return m_matrix_beam;}
        virtual void set_scalar_beam(std::shared_ptr<std::vector<float>> scalar_beam) {m_scalar_beam = scalar_beam;}
        virtual void set_matrix_beam(std::shared_ptr<std::vector<std::complex<float>>> matrix_beam);

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


        proxy::Proxy* create_proxy();

        Type m_architecture;
        proxy::Proxy* m_proxy;
        std::map<int, proxy::Proxy*> m_proxies;
        BufferSetType m_buffer_set_type;
        std::vector<std::unique_ptr<GridderBuffer>> m_gridderbuffers;
        std::vector<std::unique_ptr<DegridderBuffer>> m_degridderbuffers;
        std::vector<float> m_taper_subgrid;
        std::vector<float> m_taper_grid;
        std::vector<std::complex<float>> m_average_beam;
        std::shared_ptr<std::vector<float>> m_scalar_beam;
        std::shared_ptr<std::vector<std::complex<float>>> m_matrix_beam;
        Array4D<std::complex<float>> m_avg_aterm_correction;
        Grid m_grid;
        size_t m_subgridsize;
        float m_image_size;
        float m_cell_size;
        float m_w_step;
        size_t m_size;
        size_t m_padded_size;
        float m_kernel_size;
        float m_uv_span_time;
        float m_uv_span_frequency;
        bool m_apply_aterm = false;
        bool m_do_gridding = true;
        bool m_do_compute_avg_beam = false;

        friend BufferImpl;
        friend GridderBufferImpl;
        friend DegridderBufferImpl;

    };

} // namespace api
} // namespace idg

#endif
