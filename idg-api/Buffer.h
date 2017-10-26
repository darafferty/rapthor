/*
 * Buffer.h
 * Access to IDG's high level gridder routines
 */

#ifndef IDG_BUFFER_H_
#define IDG_BUFFER_H_

#include <cstddef>
#include <complex>
#include <utility>

namespace idg {
namespace api {    

    enum class Type {
        CPU_REFERENCE,
        CPU_OPTIMIZED,
        CUDA_GENERIC,
        OPENCL_GENERIC,
        HYBRID_CUDA_CPU_OPTIMIZED
    };


    enum class Direction {
        FourierToImage = -1,
        ImageToFourier = 1
    };

    class Buffer
    {
    public:

        virtual ~Buffer() {};

        /** \brief Signal that not more visibilies are gridded */
        virtual void finished() = 0;

        /** \brief Explicitly flush the buffer */
        virtual void flush() = 0;

    protected:
        Buffer() {}
    };

} // namespace api
} // namespace idg

#endif
