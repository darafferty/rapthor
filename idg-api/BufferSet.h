#ifndef IDG_API_BUFFERSET_H_
#define IDG_API_BUFFERSET_H_

#include <vector>
#include <string>
#include <map>
#include <memory>

#include "Value.h"
#include "GridderBuffer.h"
#include "DegridderBuffer.h"

namespace idg {
namespace api {

    enum class BufferSetType
    {
        gridding,
        degridding
    };

    enum compute_flags {
        compute_only = 1,
        compute_and_grid = 2
    };


    typedef std::map<std::string, Value> options_type;

    class BufferSet {
    public:

        static BufferSet* create(
            Type architecture);

        virtual ~BufferSet() {};

        static uint64_t get_memory_per_timestep(size_t nStations, size_t nChannels);

        virtual void init(
            size_t width,
            float cellsize,
            float max_w,
            float shiftl, float shiftm, float shiftp,
            options_type &options) = 0;

        virtual void init_buffers(
            size_t bufferTimesteps,
            std::vector<std::vector<double>> bands,
            int nr_stations,
            float max_baseline,
            options_type &options,
            BufferSetType buffer_set_type) = 0;

        virtual DegridderBuffer* get_degridder(int i) = 0;
        virtual GridderBuffer* get_gridder(int i) = 0;

        virtual void set_image(const double* image, bool do_scale = false) = 0 ;
        virtual void get_image(double* image) = 0;
        virtual void finished() = 0;

        virtual size_t get_subgridsize() = 0;
        virtual float get_subgrid_pixelsize() = 0;

        virtual void set_apply_aterm(bool do_apply) = 0;

        virtual void init_compute_avg_beam(compute_flags flag) = 0;
        virtual void finalize_compute_avg_beam() = 0;
        virtual std::shared_ptr<std::vector<float>> get_scalar_beam() const = 0;
        virtual std::shared_ptr<std::vector<std::complex<float>>> get_matrix_inverse_beam() const = 0;
        virtual void set_scalar_beam(std::shared_ptr<std::vector<float>>) = 0;
        virtual void set_matrix_inverse_beam(std::shared_ptr<std::vector<std::complex<float>>>) = 0;
        virtual void unset_matrix_inverse_beam() = 0;


    protected:
        BufferSet() {}
    };

} // namespace api
} // namespace idg

#endif
