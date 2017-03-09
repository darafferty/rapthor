#ifndef IDG_API_BUFFERSET_H_
#define IDG_API_BUFFERSET_H_

#include <vector>

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

        static BufferSet* create(
            Type architecture, 
            size_t bufferTimesteps, 
            std::vector<std::vector<double>> bands,
            int nr_stations,
            size_t width, 
            float cellsize, 
            float max_w,         
            BufferSetType buffer_set_type);

        virtual ~BufferSet() {};

        virtual DegridderBuffer* get_degridder(int i) = 0;
        virtual GridderBuffer* get_gridder(int i) = 0;

        virtual void set_image(const double* image) = 0 ;
        virtual void get_image(double* image) = 0;
        virtual void finished() = 0;
    protected:
        BufferSet() {}
    };

} // namespace api
} // namespace idg

#endif
