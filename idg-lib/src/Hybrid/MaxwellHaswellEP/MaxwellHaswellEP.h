/**
 *  \class Maxwell-HaswellEP
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_MAXWELLHASWELLEP_H_
#define IDG_MAXWELLHASWELLEP_H_

// TODO: check which include files are really necessary
#include <dlfcn.h>
#include <cuda.h>
#include "fftw3.h" // FFTW_BACKWARD, FFTW_FORWARD
#include "Proxy.h"
#include "HaswellEP.h"
#include "Maxwell.h"


namespace idg {
    namespace proxy {
        namespace hybrid {

        class MaxwellHaswellEP : public Proxy {

            public:
                /// Constructors
                MaxwellHaswellEP(Parameters params);

                /// Destructor
                virtual ~MaxwellHaswellEP() = default;

                /// Assignment
                MaxwellHaswellEP& operator=(const MaxwellHaswellEP& rhs) = delete;

            private:
                idg::proxy::cpu::HaswellEP xeon;
                idg::proxy::cuda::Maxwell cuda;

        }; // class MaxwellHaswellEP

        } // namespace hybrid
    } // namespace proxy
} // namespace idg

#endif
