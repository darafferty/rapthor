#include "idg-config.h"
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
                ProxyInfo info)
                : CUDA(params, deviceNumber, compiler, flags, info)
            {
                #if defined(DEBUG)
                cout << "Maxwell::" << __func__ << endl;
                cout << "Compiler: " << compiler << endl;
                cout << "Compiler flags: " << flags << endl;
                cout << params;
                #endif

                find_kernel_functions();
            }

            // TODO: this method is currently not used
            int get_jobsize(int nr_subgrids, Parameters mParams, cu::Device &device, int nr_streams) {
                // Get parameters
                auto nr_timesteps = mParams.get_nr_timesteps();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();

                // Set jobsize to match available gpu memory
                uint64_t device_memory_required = SIZEOF_VISIBILITIES + SIZEOF_UVW + SIZEOF_SUBGRIDS + SIZEOF_METADATA;
                uint64_t device_memory_available = device.free_memory();
                int jobsize = (device_memory_available * 0.7) / (device_memory_required * nr_streams);

                // Make sure that jobsize isn't too large
                int max_jobsize = nr_subgrids / 8;
                if (jobsize >= max_jobsize) {
                    jobsize = max_jobsize;
                }

                #if defined (DEBUG)
                clog << "nr_subgrids: " << nr_subgrids << endl;
                clog << "jobsize:     " << jobsize << endl;
                clog << "free size:   " << device_memory_available * 1e-9 << " Gb" << endl;
                clog << "buffersize:  " << nr_streams * jobsize * device_memory_required * 1e-9 << " Gb" << endl;
                #endif

                return jobsize;
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

                p.add_lib(libgridder);
                p.add_lib(libdegridder);
                p.add_lib(libfft);

                p.add_src_file_to_lib(libgridder, "KernelGridder.cu");
                p.add_src_file_to_lib(libdegridder, "KernelDegridder.cu");
                p.add_src_file_to_lib(libfft, "KernelFFT.cu");

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


        } // namespace cuda
    } // namespace proxy
} // namespace idg
