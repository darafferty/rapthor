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

        DegridderBuffer* get_degridder(int i) final override;
        GridderBuffer* get_gridder(int i) final override;

        virtual void init(
            size_t width,
            float cellsize,
            float max_w,
            float shiftl, float shiftm, float shiftp,
            options_type &options) final override;

        virtual void init_buffers(
            size_t bufferTimesteps,
            std::vector<std::vector<double>> bands,
            int nr_stations,
            float max_baseline,
            options_type &options,
            BufferSetType buffer_set_type) final override;

        virtual void set_image(const double* image, bool do_scale) final override;
        virtual void get_image(double* image) final override;
        virtual void finished() final override;

        virtual size_t get_subgridsize()  final override { return m_subgridsize; }
        virtual float get_subgrid_pixelsize()  final override { return m_image_size/m_subgridsize; }
        virtual void set_apply_aterm(bool do_apply)  final override { m_apply_aterm = do_apply; }
        virtual void init_compute_avg_beam(compute_flags flag) final override;
        virtual void finalize_compute_avg_beam() final override;
        virtual std::shared_ptr<std::vector<float>> get_scalar_beam() const final override {return m_scalar_beam;}
        virtual std::shared_ptr<std::vector<std::complex<float>>> get_matrix_inverse_beam() const final override {return m_matrix_inverse_beam;}
        virtual void set_scalar_beam(std::shared_ptr<std::vector<float>> scalar_beam) final override {m_scalar_beam = scalar_beam;}
        virtual void set_matrix_inverse_beam(std::shared_ptr<std::vector<std::complex<float>>> matrix_inverse_beam) final override;
        virtual void unset_matrix_inverse_beam() final override;

    private:

        proxy::Proxy* create_proxy();

        Type m_architecture;
        proxy::Proxy* m_proxy;
        BufferSetType m_buffer_set_type;
        std::vector<std::unique_ptr<GridderBuffer>> m_gridderbuffers;
        std::vector<std::unique_ptr<DegridderBuffer>> m_degridderbuffers;
        std::vector<float> m_taper_subgrid;
        std::vector<float> m_taper_grid;
        std::vector<float> m_inv_taper;
        std::vector<std::complex<float>> m_average_beam;
        std::shared_ptr<std::vector<float>> m_scalar_beam;
        std::shared_ptr<std::vector<std::complex<float>>> m_matrix_inverse_beam;
        Array4D<std::complex<float>> m_default_aterm_correction;
        Array4D<std::complex<float>> m_avg_aterm_correction;
        Grid m_grid;
        size_t m_subgridsize;
        float m_image_size;
        float m_cell_size;
        float m_w_step;
        float m_shift[3];
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
