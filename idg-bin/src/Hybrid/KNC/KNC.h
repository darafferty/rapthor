/**
 *  \class KNC
 *
 *  \brief Class for CPU using a KNC via offloading.
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
                    const int *baselines,
                    std::complex<float> *grid,
                    const float w_offset,
                    const std::complex<float> *aterms,
                    const float *spheroidal) override;

                virtual void degrid_visibilities(
                    std::complex<float> *visibilities,
                    const float *uvw,
                    const float *wavenumbers,
                    const int *baselines,
                    const std::complex<float> *grid,
                    const float w_offset,
                    const std::complex<float> *aterms,
                    const float *spheroidal) override;

                virtual void transform(DomainAtoDomainB direction,
                                       std::complex<float>* grid) override;


                // Low-level interface (see also CPU class)
                virtual void grid_onto_subgrids(
                    const unsigned nr_subgrids,
                    const float w_offset,
                    const float *uvw,
                    const float *wavenumbers,
                    const std::complex<float> *visibilities,
                    const float *spheroidal,
                    const std::complex<float> *aterms,
                    const int *metadata,
                    std::complex<float> *subgrids);

                virtual void add_subgrids_to_grid(
                    const unsigned nr_subgrids,
                    const int *metadata,
                    const std::complex<float> *subgrids,
                    std::complex<float> *grid);

                virtual void split_grid_into_subgrids(
                    const unsigned nr_subgrids,
                    const int *metadata,
                    std::complex<float> *subgrids,
                    const std::complex<float> *grid);

                virtual void degrid_from_subgrids(
                    const unsigned nr_subgrids,
                    const float w_offset,
                    const float *uvw,
                    const float *wavenumbers,
                    std::complex<float> *visibilities,
                    const float *spheroidal,
                    const std::complex<float> *aterms,
                    const int *metadata,
                    const std::complex<float> *subgrids);

        }; // class KNC

        } // namespace hybrid
    } // namespace proxy
} // namespace idg

#endif
