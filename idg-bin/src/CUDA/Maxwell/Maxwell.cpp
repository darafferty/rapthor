#include "Maxwell.h"

using namespace std;
using namespace idg::kernel::cuda;

namespace idg {
    namespace proxy {
        namespace cuda {
            /// Constructors
            Maxwell::Maxwell(
                Parameters params,
                unsigned deviceNumber,
                Compiler compiler,
                Compilerflags flags,
                ProxyInfo info,
                int max_nr_timesteps_gridder,
                int max_nr_timesteps_degridder)
                : CUDA(
                    params, deviceNumber, compiler, flags, info,
                    max_nr_timesteps_gridder,
                    max_nr_timesteps_degridder)
            {
                #if defined(DEBUG)
                cout << "Maxwell::" << __func__ << endl;
                cout << "Compiler: " << compiler << endl;
                cout << "Compiler flags: " << flags << endl;
                cout << params;
                #endif
            }

            ProxyInfo Maxwell::default_info() {
                #if defined(DEBUG)
                cout << "CUDA::" << __func__ << endl;
                #endif

                string srcdir = string(IDG_INSTALL_DIR)
                    + "/lib/kernels/CUDA/Maxwell";

                #if defined(DEBUG)
                cout << "Searching for source files in: " << srcdir << endl;
                #endif

                // Create temp directory
                string tmpdir = make_tempdir();

                // Create proxy info
                ProxyInfo p = default_proxyinfo(srcdir, tmpdir);

                return p;
            }

            ProxyInfo Maxwell::default_proxyinfo(string srcdir, string tmpdir) {
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

            unique_ptr<Gridder> Maxwell::get_kernel_gridder() const {
                return unique_ptr<Gridder>(new GridderMaxwell(*(modules[which_module.at(name_gridder)]), mParams));
            }

            unique_ptr<Degridder> Maxwell::get_kernel_degridder() const {
                return unique_ptr<Degridder>(new DegridderMaxwell(*(modules[which_module.at(name_degridder)]), mParams));
            }

            unique_ptr<GridFFT> Maxwell::get_kernel_fft() const {
                return unique_ptr<GridFFT>(new GridFFTMaxwell(*(modules[which_module.at(name_fft)]), mParams));
            }

            unique_ptr<Scaler> Maxwell::get_kernel_scaler() const {
                return unique_ptr<Scaler>(new ScalerMaxwell(*(modules[which_module.at(name_scaler)]), mParams));
            }

            unique_ptr<Adder> Maxwell::get_kernel_adder() const {
                return unique_ptr<Adder>(new AdderMaxwell(*(modules[which_module.at(name_adder)]), mParams));
            }

            unique_ptr<Splitter> Maxwell::get_kernel_splitter() const {
                return unique_ptr<Splitter>(new SplitterMaxwell(*(modules[which_module.at(name_splitter)]), mParams));
            }
        } // namespace cuda
    } // namespace proxy
} // namespace idg
