/**
 *  \class Reference
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_CUDA_REFERENCE_H_
#define IDG_CUDA_REFERENCE_H_

#include "CUDA.h"

namespace idg {
    namespace proxy {
        namespace cuda {

            class Reference : public CUDA {
            
            public:
                /// Constructors
                Reference(Parameters params,
                          unsigned deviceNumber = 0,
                          Compiler compiler = default_compiler(),
                          Compilerflags flags = default_compiler_flags(),
                          ProxyInfo info = default_info());
                
                /// Destructor
                ~Reference() = default;
    
                // Get default values for ProxyInfo
                static ProxyInfo default_info();
                static std::string default_compiler();
                static std::string default_compiler_flags();

         }; // class Reference
    
        } // namespace cpu
    } // namespace proxy
} // namespace idg

#endif
