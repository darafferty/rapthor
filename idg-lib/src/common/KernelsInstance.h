#ifndef IDG_KERNELS_H_
#define IDG_KERNELS_H_

#include <cassert>

#include "idg-common.h"


namespace idg {
    namespace kernel {

        class KernelsInstance
        {
            public:
                /*
                    Misc math routines
                */
                void shift(
                    Array3D<std::complex<float>>& data) const;

                void scale(
                    Array3D<std::complex<float>>& data,
                    std::complex<float> scale) const;

                void tile_backward(
                    const int tile_size,
                    const Grid& grid_src,
                          Grid& grid_dst) const;

                void tile_forward(
                    const int tile_size,
                    const Grid& grid_src,
                          Grid& grid_dst) const;

        }; // end class KernelsInstance

    } // namespace kernel
} // namespace idg
#endif
