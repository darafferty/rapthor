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

        class KNCOffload : public CPU {

            public:
                /// Constructors
                KNCOffload(Compiler compiler,
                          Compilerflags flags,
                          Parameters params,
                          ProxyInfo info = default_info());
                
                /// Destructor
                ~KNCOffload() = default;
    
                // Get default values for ProxyInfo
                static ProxyInfo default_info();
                static std::string default_compiler();
                static std::string default_compiler_flags();
            
            protected:
                virtual void run_gridder(int jobsize, GRIDDER_PARAMETERS);
        
                virtual void run_adder(int jobsize, ADDER_PARAMETERS);
        
                virtual void run_splitter(int jobsize, SPLITTER_PARAMETERS);
        
                virtual void run_degridder(int jobsize, DEGRIDDER_PARAMETERS);
        
                virtual void run_fft(FFT_PARAMETERS);

            private:
                void run_gridder_intel_leo(int jobsize, GRIDDER_PARAMETERS);
                void run_gridder_omp4(int jobsize, GRIDDER_PARAMETERS);

        
        }; // class HaswellEP
    
    } // namespace proxy
} // namespace idg

#endif
