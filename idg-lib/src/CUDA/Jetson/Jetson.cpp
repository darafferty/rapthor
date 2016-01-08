#include "idg-config.h"
#include "Jetson.h"

using namespace std;
using namespace idg::kernel::cuda;

namespace idg {
    namespace proxy {
        namespace cuda {
            /// Constructors
            Jetson::Jetson(
                Parameters params,
                unsigned deviceNumber,
                Compiler compiler,
                Compilerflags flags,
                ProxyInfo info)
                : CUDA(params, deviceNumber, compiler, flags, info)
            {
                #if defined(DEBUG)
                cout << "Jetson::" << __func__ << endl;
                cout << "Compiler: " << compiler << endl;
                cout << "Compiler flags: " << flags << endl;
                cout << params;
                #endif

                find_kernel_functions();
            }

            ProxyInfo Jetson::default_info() {
                #if defined(DEBUG)
                cout << "CUDA::" << __func__ << endl;
                #endif

                string srcdir = string(IDG_INSTALL_DIR)
                    + "/lib/kernels/CUDA/Jetson";

                #if defined(DEBUG)
                cout << "Searching for source files in: " << srcdir << endl;
                #endif

                // Create temp directory
                string tmpdir = make_tempdir();

                // Create proxy info
                ProxyInfo p = default_proxyinfo(srcdir, tmpdir);

                return p;
            }

            ProxyInfo Jetson::default_proxyinfo(string srcdir, string tmpdir) {
                ProxyInfo p;
                p.set_path_to_src(srcdir);
                p.set_path_to_lib(tmpdir);

                string libgridder = "Gridder.ptx";
                string libdegridder = "Degridder.ptx";
                string libfft = "FFT.ptx";
                string libscaler = "Scaler.ptx";
                string libadder = "Adder.ptx";
                string libsplitter = "Splitter.ptx";

                p.add_lib(libgridder);
                p.add_lib(libdegridder);
                p.add_lib(libfft);
                p.add_lib(libscaler);
                p.add_lib(libadder);
                p.add_lib(libsplitter);

                p.add_src_file_to_lib(libgridder, "KernelGridder.cu");
                p.add_src_file_to_lib(libdegridder, "KernelDegridder.cu");
                p.add_src_file_to_lib(libfft, "KernelFFT.cu");
                p.add_src_file_to_lib(libscaler, "KernelScaler.cu");
                p.add_src_file_to_lib(libadder, "KernelAdder.cu");
                p.add_src_file_to_lib(libsplitter, "KernelSplitter.cu");

                p.set_delete_shared_objects(true);

                return p;
            }

#if 1
            void Jetson::transform(DomainAtoDomainB direction,
                                complex<float>* grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                cout << "Transform direction: " << direction << endl;
                #endif

                // Constants
                auto gridsize = mParams.get_grid_size();
                auto nr_polarizations = mParams.get_nr_polarizations();

                cu::HostMemory h_grid(SIZEOF_GRID);
                cu::DeviceMemory d_grid(h_grid);
                //transform(direction, context, h_grid);
            }

            void Jetson::grid_visibilities(
                const complex<float> *visibilities,
                const float *uvw,
                const float *wavenumbers,
                const int *baselines,
                complex<float> *grid,
                const float w_offset,
                const complex<float> *aterm,
                const float *spheroidal)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
                cout << "Not implemented" << endl;
            }

            void Jetson::degrid_visibilities(
                std::complex<float> *visibilities,
                const float *uvw,
                const float *wavenumbers,
                const int *baselines,
                const std::complex<float> *grid,
                const float w_offset,
                const std::complex<float> *aterm,
                const float *spheroidal)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
                cout << "Not implemented" << endl;
            }
#endif
 
            unique_ptr<Gridder> Jetson::get_kernel_gridder() const {
                return unique_ptr<Gridder>(new GridderJetson(*(modules[which_module.at(name_gridder)]), mParams));
            }

            unique_ptr<Degridder> Jetson::get_kernel_degridder() const {
                return unique_ptr<Degridder>(new DegridderJetson(*(modules[which_module.at(name_degridder)]), mParams));
            }

            unique_ptr<GridFFT> Jetson::get_kernel_fft() const {
                return unique_ptr<GridFFT>(new GridFFTJetson(*(modules[which_module.at(name_fft)]), mParams));
            }

            unique_ptr<Scaler> Jetson::get_kernel_scaler() const {
                return unique_ptr<Scaler>(new ScalerJetson(*(modules[which_module.at(name_scaler)]), mParams));
            }

            unique_ptr<Adder> Jetson::get_kernel_adder() const {
                return unique_ptr<Adder>(new AdderJetson(*(modules[which_module.at(name_adder)]), mParams));
            }

            unique_ptr<Splitter> Jetson::get_kernel_splitter() const {
                return unique_ptr<Splitter>(new SplitterJetson(*(modules[which_module.at(name_splitter)]), mParams));
            }
        } // namespace cuda
    } // namespace proxy
} // namespace idg
