/*
 * DegridderPlan.h
 * Access to IDG's high level gridder routines
 */

#ifndef IDG_DEGRIDDERPLAN_H_
#define IDG_DEGRIDDERPLAN_H_

#include <complex>
#include <vector>
#include <set>
#include <algorithm>
#include <utility>
#include <map>
#include <stdexcept>
#include <cmath>

#include "idg-common.h"
#if defined(BUILD_LIB_CPU)
#include "idg-cpu.h"
#endif
#include "Scheme.h"

namespace idg {

    class DegridderPlan : public Scheme
    {
    public:
        // Constructors and destructor
        DegridderPlan(Type architecture = Type::CPU_REFERENCE,
                      size_t bufferTimesteps = 4096);

        virtual ~DegridderPlan();

        void request_visibilities(
            size_t timeIndex,                  // 0 <= timeIndex < NR_TIMESTEPS
            size_t antenna1,                   // 0 <= antenna1 < nrStations
            size_t antenna2,                   // antenna1 < antenna2 < nrStations
            const double* uvwInMeters);        // (u, v, w)

        void read_visibilities(
            size_t timeIndex,                   // 0 <= timeIndex < NR_TIMESTEPS
            size_t antenna1,                    // 0 <= antenna1 < nrStations
            size_t antenna2,                    // antenna1 < antenna2 < nrStations
            std::complex<float>* visibilities); // size CH x PL

        // To flush the buffer explicitly
        virtual void flush() override;

        // To transform the gridded visibilities to an image
        virtual void transform_grid(
            double crop_tolerance      = 5e-3,
            size_t nr_polarizations    = 0,
            size_t height              = 0,
            size_t width               = 0,
            std::complex<double> *grid = nullptr) override;
    };

} // namespace idg

#endif
