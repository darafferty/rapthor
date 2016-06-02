/*
 * GridderPlan.h
 * Access to IDG's high level gridder routines
 */

#ifndef IDG_GRIDDERPLAN_H_
#define IDG_GRIDDERPLAN_H_

#include <complex>
#include <vector>
#include <set>
#include <algorithm>
#include <stdexcept>
#include <cmath>

#include "idg-common.h"
#if defined(BUILD_LIB_CPU)
#include "idg-cpu.h"
#endif
#include "Scheme.h"

namespace idg {

    class GridderPlan : public Scheme
    {
    public:
        // Constructors and destructor
        GridderPlan(Type architecture = Type::CPU_REFERENCE,
                    size_t bufferTimesteps = 4096);

        virtual ~GridderPlan();

        // Adds the visibilities to the buffer and eventually to the grid
        void grid_visibilities(
            const std::complex<float>* visibilities, // size CH x PL
            const double* uvwInMeters,               // (u, v, w)
            size_t antenna1,                         // 0 <= antenna1 < nrStations
            size_t antenna2,                         // antenna1 < antenna2 < nrStations
            size_t timeIndex);                       // 0 <= timeIndex < NR_TIMESTEPS

        // To flush the buffer explicitly
        virtual void flush() override;

        // To transform the provided image before prediction
        virtual void transform_grid(std::complex<double> *grid = nullptr) override;
    };

} // namespace idg

#endif
