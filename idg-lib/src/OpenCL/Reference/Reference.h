/**
 *  \class Reference
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_OPENCL_REFERENCE_H_
#define IDG_OPENCL_REFERENCE_H_

#include "OpenCL.h"

namespace idg {
    namespace proxy {
        namespace opencl {

            class Reference : public OpenCL {
            
            public:
                /// Constructors
                Reference(Parameters params,
                          unsigned deviceNumber = 0,
                          Compilerflags flags = default_compiler_flags(),
                          ProxyInfo info = default_info());
                
                /// Destructor
                ~Reference() = default;
    
                // Get default values for ProxyInfo
                static ProxyInfo default_info();
                static std::string default_compiler_flags();
                    
            }; // class Reference
    
        } // namespace opencl
    } // namespace proxy
} // namespace idg

#endif
