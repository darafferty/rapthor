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

#include "idg-common.h"
#include "Kernels.h"

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

                // High level interface, inherited from Proxy
                virtual void grid_visibilities(
                    const std::complex<float> *visibilities,
                    const float *uvw,
                    const float *wavenumbers,
                    const int *metadata,
                    std::complex<float> *grid,
                    const float w_offset,
                    const std::complex<float> *aterms,
                    const float *spheroidal) override;

                virtual void degrid_visibilities(
                    std::complex<float> *visibilities,
                    const float *uvw,
                    const float *wavenumbers,
                    const int *metadata,
                    const std::complex<float> *grid,
                    const float w_offset,
                    const std::complex<float> *aterms,
                    const float *spheroidal) override;

                virtual void transform(DomainAtoDomainB direction,
                                       std::complex<float>* grid) override;


                // Low-level interface (see also CPU class)
                virtual void grid_onto_subgrids(
                    unsigned nr_subgrids,
                    float w_offset,
                    float *uvw,
                    float *wavenumbers,
                    std::complex<float> *visibilities,
                    float *spheroidal,
                    std::complex<float> *aterms,
                    int *metadata,
                    std::complex<float> *subgrids);

                virtual void add_subgrids_to_grid(
                    unsigned nr_subgrids,
                    int *metadata,
                    std::complex<float> *subgrids,
                    std::complex<float> *grid);

                virtual void split_grid_into_subgrids(
                    unsigned nr_subgrids,
                    int *metadata,
                    std::complex<float> *subgrids,
                    std::complex<float> *grid);

                virtual void degrid_from_subgrids(
                    unsigned nr_subgrids,
                    float w_offset,
                    float *uvw,
                    float *wavenumbers,
                    std::complex<float> *visibilities,
                    float *spheroidal,
                    std::complex<float> *aterms,
                    int *metadata,
                    std::complex<float> *subgrids);


                //protected:
                //void run_gridder(int jobsize, GRIDDER_PARAMETERS);
                //void run_adder(int jobsize, ADDER_PARAMETERS);
                //void run_splitter(int jobsize, SPLITTER_PARAMETERS);
                //void run_degridder(int jobsize, DEGRIDDER_PARAMETERS);
                //void run_fft(FFT_PARAMETERS);

        }; // class KNC

        } // namespace hybrid
    } // namespace proxy
} // namespace idg

#endif
