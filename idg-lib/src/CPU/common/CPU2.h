#ifndef IDG_CPU2_H_
#define IDG_CPU2_H_

#include "idg-common.h"

namespace idg {
    namespace proxy {
        namespace cpu {

            class CPU2 : public Proxy2
            {
                public:
                    // Constructor
                    CPU2(
                        Compiler compiler,
                        Compilerflags flags,
                        ProxyInfo info);

                    // Disallow assignment and pass-by-value
                    CPU2& operator=(const CPU2& rhs) = delete;
                    CPU2(const CPU2& v) = delete;

                    // Destructor
                    virtual ~CPU2();

                    // Routines
                    virtual void gridding(
                        const float w_offset, // in lambda
                        const unsigned int kernel_size, // full width in pixels
                        const Array1D<float>& frequencies, // TODO: convert from wavenumbers
                        const Array3D<Visibility<std::complex<float>>>& visibilities,
                        const Array2D<UVWCoordinate<float>>& uvw,
                        const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                        Array3D<std::complex<float>>& grid,
                        const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                        const Array1D<unsigned int>& aterms_offsets,
                        const Array2D<float>& spheroidal) override;

                    virtual void degridding(
                        const float w_offset, // in lambda
                        const unsigned int kernel_size, // full width in pixels
                        const Array1D<float>& frequencies, // TODO: convert from wavenumbers
                        Array3D<Visibility<std::complex<float>>>& visibilities,
                        const Array2D<UVWCoordinate<float>>& uvw,
                        const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                        const Array3D<std::complex<float>>& grid,
                        const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                        const Array1D<unsigned int>& aterms_offsets,
                        const Array2D<float>& spheroidal) override;

                    virtual void transform(
                        DomainAtoDomainB direction,
                        const Array3D<std::complex<float>>& grid) override;

                protected:
                    static std::string make_tempdir();
                    static ProxyInfo default_proxyinfo(
                        std::string srcdir,
                        std::string tmpdir);

            }; // end class CPU2

        } // end namespace cpu
    } // end namespace proxy
} // end namespace idg
#endif
