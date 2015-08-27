/**
 *  \class KNCOffload
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_KNCOFFLOAD_H_
#define IDG_KNCOFFLOAD_H_

// TODO: check which include files are really necessary
#include <dlfcn.h>
#include "fftw3.h" // FFTW_BACKWARD, FFTW_FORWARD
#include "AbstractProxy.h"

namespace idg {

    namespace proxy {

        class KNCOffload : public Proxy {

            public:
                /// Constructors
                KNCOffload(Parameters params);
                
                /// Destructor
                ~KNCOffload() = default;
    
            public:
                void grid_onto_subgrids(int jobsize, GRIDDER_PARAMETERS);
                void add_subgrids_to_grid(int jobsize, ADDER_PARAMETERS);
                void split_grid_into_subgrids(int jobsize, SPLITTER_PARAMETERS);
                void degrid_from_subgrids(int jobsize, DEGRIDDER_PARAMETERS);
                void transform(DomainAtoDomainB direction, void* grid);

            protected:
                void run_gridder(int jobsize, GRIDDER_PARAMETERS);
                void run_adder(int jobsize, ADDER_PARAMETERS);
                void run_splitter(int jobsize, SPLITTER_PARAMETERS);
                void run_degridder(int jobsize, DEGRIDDER_PARAMETERS);
                void run_fft(FFT_PARAMETERS);
        
        }; // class KNCOffload
    
    } // namespace proxy
} // namespace idg

#endif
