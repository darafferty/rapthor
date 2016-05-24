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

        virtual void request_visibilities(
            size_t rowId,
            const double* uvwInMeters,
            size_t antenna1,
            size_t antenna2,
            size_t timeIndex);

        virtual void read_visibilities(
            size_t rowId,
            std::complex<float>* visibilities) const;

        virtual void read_visibilities(
            size_t antenna1,
            size_t antenna2,
            size_t timeIndex,
            std::complex<float>* visibilities) const;

        // Must be called to flush the buffer
        virtual void flush() override;

    private:
        std::map<size_t,std::pair<size_t,size_t>> m_rowid_to_bufferindex;
    };

} // namespace idg

#endif
