#include "idg-config.h"
#include "Kepler.h"

using namespace std;

namespace idg {
    namespace proxy {
        namespace cuda {
            /*
                Power measurement
            */
            static PowerSensor *powerSensor;

            class PowerRecord {
                public:
                    void enqueue(cu::Stream &stream);
                    static void getPower(CUstream, CUresult, void *userData);
                    PowerSensor::State state;
                    cu::Event event;
            };
            
            void PowerRecord::enqueue(cu::Stream &stream) {
                stream.record(event);
                stream.addCallback((CUstreamCallback) &PowerRecord::getPower, &state);
            }
            
            void PowerRecord::getPower(CUstream, CUresult, void *userData) {
                *static_cast<PowerSensor::State *>(userData) = powerSensor->read();
            }

            /// Constructors
            Kepler::Kepler(
                Parameters params,
                unsigned deviceNumber,
                Compiler compiler,
                Compilerflags flags,
                ProxyInfo info)
                : CUDA(params, deviceNumber, compiler, flags, info)
            {
                #if defined(DEBUG)
                cout << "Kepler::" << __func__ << endl;
                cout << "Compiler: " << compiler << endl;
                cout << "Compiler flags: " << flags << endl;
                cout << params;
                #endif

                find_kernel_functions();
            }

            ProxyInfo Kepler::default_info() {
                #if defined(DEBUG)
                cout << "CUDA::" << __func__ << endl;
                #endif

                string srcdir = string(IDG_SOURCE_DIR)
                    + "/src/CUDA/Kepler/kernels";

                #if defined(DEBUG)
                cout << "Searching for source files in: " << srcdir << endl;
                #endif

                // Create temp directory
                string tmpdir = make_tempdir();

                // Create proxy info
                ProxyInfo p = default_proxyinfo(srcdir, tmpdir);

                return p;
            }

            ProxyInfo Kepler::default_proxyinfo(string srcdir, string tmpdir) {
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

            void Kepler::find_kernel_functions() {
                #if defined(DEBUG)
                cout << "Kepler::" << __func__ << endl;
                #endif

                CUfunction function;
                for (unsigned int i=0; i<modules.size(); i++) {
                    if (cuModuleGetFunction(&function, *modules[i], kernel::name_gridder.c_str()) == CUDA_SUCCESS) {
                        // found gridder kernel in module i
                        which_module[kernel::name_gridder] = i;
                    }
                    if (cuModuleGetFunction(&function, *modules[i], kernel::name_degridder.c_str()) == CUDA_SUCCESS) {
                        // found degridder kernel in module i
                        which_module[kernel::name_degridder] = i;
                    }
                    if (cuModuleGetFunction(&function, *modules[i], kernel::name_fft.c_str()) == CUDA_SUCCESS) {
                        // found fft kernel in module i
                        which_module[kernel::name_fft] = i;
                    }
                } // end for
            } // end find_kernel_functions

        } // namespace cuda
    } // namespace proxy
} // namespace idg
