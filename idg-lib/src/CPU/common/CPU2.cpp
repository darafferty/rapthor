#include "CPU2.h"

using namespace std;
using namespace idg;

namespace idg {
    namespace proxy {
        namespace cpu {

            // Constructor
            CPU2::CPU2(
                Compiler compiler,
                Compilerflags flags,
                ProxyInfo info)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                cout << "Compiler: " << compiler << endl;
                cout << "Compiler flags: " << flags << endl;
                #endif

                //compile(compiler, flags);
                //load_shared_objects();
                //find_kernel_functions();
            }

            // Destructor
            CPU2::~CPU2()
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Unload shared objects by ~Module
                //for (unsigned int i = 0; i < modules.size(); i++) {
                //    delete modules[i];
                //}

                // Delete .so files
                //if (mInfo.delete_shared_objects()) {
                //    for (auto libname : mInfo.get_lib_names()) {
                //        string lib = mInfo.get_path_to_lib() + "/" + libname;
                //        remove(lib.c_str());
                //    }
                //    rmdir(mInfo.get_path_to_lib().c_str());
                //}
            }

            // Routines
            void CPU2::gridding(
                const float w_offset, // in lambda
                const unsigned int kernel_size, // full width in pixels
                const Array1D<float>& frequencies, // TODO: convert from wavenumbers
                const Array3D<Visibility<std::complex<float>>>& visibilities,
                const Array2D<UVWCoordinate<float>>& uvw,
                const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                Array3D<std::complex<float>>& grid,
                const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                const Array1D<unsigned int>& aterms_offsets,
                const Array2D<float>& spheroidal)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
            }

            void CPU2::degridding(
                const float w_offset, // in lambda
                const unsigned int kernel_size, // full width in pixels
                const Array1D<float>& frequencies, // TODO: convert from wavenumbers
                Array3D<Visibility<std::complex<float>>>& visibilities,
                const Array2D<UVWCoordinate<float>>& uvw,
                const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                const Array3D<std::complex<float>>& grid,
                const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                const Array1D<unsigned int>& aterms_offsets,
                const Array2D<float>& spheroidal)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
            }

            void CPU2::transform(
                DomainAtoDomainB direction,
                const Array3D<std::complex<float>>& grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
            }



            // Runtime compilation
            string CPU2::make_tempdir()
            {
                char _tmpdir[] = "/tmp/idg-XXXXXX";
                char *tmpdir = mkdtemp(_tmpdir);

                if (tmpdir == NULL) {
                    throw runtime_error("Cannot create tmp directory");
                }

                #if defined(DEBUG)
                cout << "Temporary files will be stored in: " << tmpdir << endl;
                #endif
                return tmpdir;
            }

            ProxyInfo CPU2::default_proxyinfo(string srcdir, string tmpdir)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                ProxyInfo p;
                p.set_path_to_src(srcdir);
                p.set_path_to_lib(tmpdir);

                string libgridder = "Gridder.so";
                string libdegridder = "Degridder.so";
                string libfft = "FFT.so";
                string libadder = "Adder.so";
                string libsplitter = "Splitter.so";

                p.add_lib(libgridder);
                p.add_lib(libdegridder);
                p.add_lib(libfft);
                p.add_lib(libadder);
                p.add_lib(libsplitter);

                p.add_src_file_to_lib(libgridder, "KernelGridder.cpp");
                p.add_src_file_to_lib(libdegridder, "KernelDegridder.cpp");
                p.add_src_file_to_lib(libfft, "KernelFFT.cpp");
                p.add_src_file_to_lib(libadder, "KernelAdder.cpp");
                p.add_src_file_to_lib(libsplitter, "KernelSplitter.cpp");

                p.set_delete_shared_objects(true);

                return p;
            }


        } // namespace cpu
    } // namespace proxy
} // namespace idg
