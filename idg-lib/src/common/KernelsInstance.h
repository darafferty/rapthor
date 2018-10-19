#ifndef IDG_KERNELS_H_
#define IDG_KERNELS_H_

#include <cassert>

#ifndef NDEBUG
#define ASSERT(x) assert(x)
#else
#define ASSERT(x) ((void)(x))
#endif

#include "Report.h"

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
                    Array3D<std::complex<float>>& data);

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

                /*
                    Performance reporting
                */
            public:
                void set_report(Report& report_) { report = &report_; }

            protected:
                Report* report = NULL;
                powersensor::PowerSensor* powerSensor;

        }; // end class KernelsInstance

    } // namespace kernel
} // namespace idg

#endif
