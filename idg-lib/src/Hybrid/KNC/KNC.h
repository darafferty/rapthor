/**
 *  \class KNC
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_HYBRID_KNC_H_
#define IDG_HYBRID_KNC_H_

#include "fftw3.h" // FFTW_BACKWARD, FFTW_FORWARD
#include "Proxy.h"
#include "PowerSensor.h"

namespace idg {
    namespace proxy {
        namespace hybrid {

        class KNC : public Proxy {

            public:
                /// Constructors
                KNC(Parameters params);

                /// Copy constructor
                KNC(const KNC& v) = delete;

                /// Destructor
                virtual ~KNC() = default;

                /// Assignment
                KNC& operator=(const KNC& rhs) = delete;

            public:
                virtual void grid_onto_subgrids(int jobsize, GRIDDER_PARAMETERS) override;
                virtual void add_subgrids_to_grid(int jobsize, ADDER_PARAMETERS) override;
                virtual void split_grid_into_subgrids(int jobsize, SPLITTER_PARAMETERS) override;
                virtual void degrid_from_subgrids(int jobsize, DEGRIDDER_PARAMETERS) override;
                virtual void transform(DomainAtoDomainB direction, void* grid) override;

            protected:
                void run_gridder(int jobsize, GRIDDER_PARAMETERS);
                void run_adder(int jobsize, ADDER_PARAMETERS);
                void run_splitter(int jobsize, SPLITTER_PARAMETERS);
                void run_degridder(int jobsize, DEGRIDDER_PARAMETERS);
                void run_fft(FFT_PARAMETERS);

        }; // class KNC

        } // namespace hybrid
    } // namespace proxy
} // namespace idg

#endif
