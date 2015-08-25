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
#include "CPU.h"
#include "Kernels.h"

namespace idg {

    namespace proxy {

        class HaswellEP : public CPU {

            public:
                /// Constructors
                HaswellEP(Compiler compiler,
                          Compilerflags flags,
                          Parameters params,
                          ProxyInfo info = default_info());
                
                // HaswellEP(CompilerEnvironment cc,
                //           Parameters params,
                //           ProxyInfo info = default_info());
    
                /// Destructor
                ~HaswellEP() = default;
    
                // Get default values for ProxyInfo
                static ProxyInfo default_info();
                static std::string default_compiler();
                static std::string default_compiler_flags();
                    
        }; // class HaswellEP
    
    } // namespace proxy
} // namespace idg

#endif
