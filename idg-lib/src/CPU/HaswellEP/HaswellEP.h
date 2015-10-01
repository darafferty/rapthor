/**
 *  \class HaswellEP
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_HASWELLEP_H_
#define IDG_HASWELLEP_H_

// TODO: check which include files are really necessary
#include <dlfcn.h>
#include "fftw3.h" // FFTW_BACKWARD, FFTW_FORWARD
#include "AbstractProxy.h"
#include "../common/CPU.h"
#include "../common/Kernels.h"

namespace idg {
    namespace proxy {
        namespace cpu {

        class HaswellEP : public CPU {

            public:
                /// Constructors
                HaswellEP(Parameters params,
                          Compiler compiler = default_compiler(),
                          Compilerflags flags = default_compiler_flags(),
                          ProxyInfo info = default_info());
                
                /// Destructor
                ~HaswellEP() = default;
    
                // Get default values for ProxyInfo
                static ProxyInfo default_info();
                static std::string default_compiler();
                static std::string default_compiler_flags();
                    
        }; // class HaswellEP
    
        } // namespace cpu
    } // namespace proxy
} // namespace idg

#endif
