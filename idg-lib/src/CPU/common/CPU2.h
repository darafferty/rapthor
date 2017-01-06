#ifndef IDG_CPU2_H_
#define IDG_CPU2_H_

#include "idg-common.h"
#include "idg-powersensor.h"

#include "Kernels.h"

// Forward declarations, TODO: remove
#define FFTW_FORWARD  (-1)
#define FFTW_BACKWARD (+1)

namespace idg {
    namespace proxy {
        namespace cpu {

            class CPU2 : public Proxy2
            {
                public:
                    // Constructor
                    CPU2(
                        CompileConstants constants,
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
                        const float cell_size,
                        const unsigned int kernel_size, // full width in pixels
                        const Array1D<float>& frequencies,
                        const Array3D<Visibility<std::complex<float>>>& visibilities,
                        const Array2D<UVWCoordinate<float>>& uvw,
                        const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                        Array3D<std::complex<float>>& grid,
                        const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                        const Array1D<unsigned int>& aterms_offsets,
                        const Array2D<float>& spheroidal) override;

                    virtual void degridding(
                        const float w_offset, // in lambda
                        const float cell_size,
                        const unsigned int kernel_size, // full width in pixels
                        const Array1D<float>& frequencies,
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

                    virtual std::unique_ptr<idg::kernel::cpu::Gridder> get_kernel_gridder() const;
                    virtual std::unique_ptr<idg::kernel::cpu::Degridder> get_kernel_degridder() const;
                    virtual std::unique_ptr<idg::kernel::cpu::Adder> get_kernel_adder() const;
                    virtual std::unique_ptr<idg::kernel::cpu::Splitter> get_kernel_splitter() const;
                    virtual std::unique_ptr<idg::kernel::cpu::GridFFT> get_kernel_fft() const;

                private:
                    void grid_onto_subgrids(
                        const Plan2& plan,
                        const float w_offset,
                        const float *uvw,
                        const float *wavenumbers,
                        const std::complex<float> *visibilities,
                        const float *spheroidal,
                        const std::complex<float> *aterm,
                        std::complex<float> *subgrids);

                    virtual void add_subgrids_to_grid(
                        const Plan2& plan,
                        const std::complex<float> *subgrids,
                        std::complex<float> *grid);


                protected:
                    static std::string make_tempdir();
                    static ProxyInfo default_proxyinfo(
                        std::string srcdir,
                        std::string tmpdir);
                    void compile();
                    void load_shared_objects();
                    void find_kernel_functions();

                    Compiler mCompiler;
                    Compilerflags mFlags;
                    ProxyInfo mInfo;
                    Parameters mParams; // TODO: remove

                    // store the ptr to Module, which each loads an .so-file
                    std::vector<runtime::Module*> modules;
                    std::map<std::string,int> which_module;

                    PowerSensor *powerSensor;
            }; // end class CPU2

        } // end namespace cpu
    } // end namespace proxy
} // end namespace idg
#endif
