/**
 *  \class HaswellEP
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_CPU_HASWELLEP_H_
#define IDG_CPU_HASWELLEP_H_

#include "idg-cpu.h"

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
                
                /// Copy constructor
                ///HaswellEP(const HaswellEP& v) = delete;

                /// Destructor
                virtual ~HaswellEP() = default;
    
                /// Assignment
                HaswellEP& operator=(const HaswellEP& rhs) = delete;

                // Get default values for ProxyInfo
                static ProxyInfo default_info();
                static std::string default_compiler();
                static std::string default_compiler_flags();
                    
        }; // class HaswellEP
    
        } // namespace cpu
    } // namespace proxy
} // namespace idg

#endif
