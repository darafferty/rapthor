#include <cstdint> // unint64_t

#include "idg-config.h"

#include "InstanceCPU.h"

using namespace std;

namespace idg {
    namespace kernel {
        namespace cpu {

            // Constructor
            InstanceCPU::InstanceCPU(
                CompileConstants constants,
                Compiler compiler,
                Compilerflags flags,
                ProxyInfo info) :
                KernelsInstance(constants),
                mInfo(info)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                cout << "Compiler: " << compiler << endl;
                cout << "Compiler flags: " << flags << endl;
                #endif

                compile(compiler, flags);
                load_shared_objects();
                find_kernel_functions();
            }

            // Destructor
            InstanceCPU::~InstanceCPU()
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Unload shared objects by ~Module
                for (unsigned int i = 0; i < modules.size(); i++) {
                    delete modules[i];
                }

                // Delete .so files
                if (mInfo.delete_shared_objects()) {
                    for (auto libname : mInfo.get_lib_names()) {
                        string lib = mInfo.get_path_to_lib() + "/" + libname;
                        remove(lib.c_str());
                    }
                    rmdir(mInfo.get_path_to_lib().c_str());
                }
            }

            unique_ptr<kernel::cpu::Gridder> InstanceCPU::get_kernel_gridder() const {
                return unique_ptr<kernel::cpu::Gridder>(
                    new kernel::cpu::Gridder(
                        *(modules[which_module.at(kernel::cpu::name_gridder)])));
            }


            unique_ptr<kernel::cpu::Degridder> InstanceCPU::get_kernel_degridder() const {
                return unique_ptr<kernel::cpu::Degridder>(
                    new kernel::cpu::Degridder(
                        *(modules[which_module.at(kernel::cpu::name_degridder)])));
            }


            unique_ptr<kernel::cpu::Adder> InstanceCPU::get_kernel_adder() const {
                return unique_ptr<kernel::cpu::Adder>(
                    new kernel::cpu::Adder(
                        *(modules[which_module.at(kernel::cpu::name_adder)])));
            }

            unique_ptr<kernel::cpu::Splitter> InstanceCPU::get_kernel_splitter() const {
                return unique_ptr<kernel::cpu::Splitter>(
                    new kernel::cpu::Splitter(
                        *(modules[which_module.at(kernel::cpu::name_splitter)])));
            }


            unique_ptr<kernel::cpu::GridFFT> InstanceCPU::get_kernel_fft() const {
                return unique_ptr<kernel::cpu::GridFFT>(
                    new kernel::cpu::GridFFT(
                        *(modules[which_module.at(kernel::cpu::name_fft)])));
            }

            string InstanceCPU::make_tempdir()
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

            ProxyInfo InstanceCPU::default_proxyinfo(
                string srcdir,
                string tmpdir)
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

            void InstanceCPU::compile(
                Compiler compiler,
                Compilerflags flags)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Set compile arguments:
                stringstream arguments;
                arguments << "-DNR_POLARIZATIONS=" << mConstants.get_nr_correlations();
                arguments << " -DSUBGRIDSIZE=" << mConstants.get_subgrid_size();
                arguments << " " << flags;
                #if defined(HAVE_MKL)
                arguments << " -DHAVE_MKL";
                arguments << " -I" << MKL_INCLUDE_DIRS;
                #endif

                // Get list of libraries to build
                vector<string> v = mInfo.get_lib_names();

                // Build all libraries
                #pragma omp parallel for num_threads(v.size())
                for (int i = 0; i < v.size(); i++) {
                    string libname = mInfo.get_lib_names()[i];

                    // create shared object "libname"
                    string lib = mInfo.get_path_to_lib() + "/" + libname;

                    vector<string> source_files = mInfo.get_source_files(libname);

                    stringstream source;
                    for (auto src : source_files) {
                        source << mInfo.get_path_to_src() << "/" << src << " ";
                    } // source = a.cpp b.cpp c.cpp ...

                    #if defined(DEBUG)
                    cout << lib << " " << source.str() << " " << endl;
                    #endif

                    runtime::Source(source.str().c_str()).compile(
                        compiler.c_str(),
                        lib.c_str(),
                        arguments.str().c_str());
                } // end for each library
            } // end compile

            void InstanceCPU::load_shared_objects()
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                for (auto libname : mInfo.get_lib_names()) {
                    string lib = mInfo.get_path_to_lib() + "/" + libname;

                    #if defined(DEBUG)
                    cout << "Loading: " << libname << endl;
                    #endif

                    modules.push_back(new runtime::Module(lib.c_str()));
                }
            } // end load_shared_objects

            // maps name -> index in modules that contain that symbol
            void InstanceCPU::find_kernel_functions()
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                for (unsigned int i=0; i<modules.size(); i++) {
                    if (dlsym(*modules[i], kernel::cpu::name_gridder.c_str())) {
                      // found gridder kernel in module i
                      which_module[kernel::cpu::name_gridder] = i;
                    }
                    if (dlsym(*modules[i], kernel::cpu::name_degridder.c_str())) {
                      // found degridder kernel in module i
                      which_module[kernel::cpu::name_degridder] = i;
                    }
                    if (dlsym(*modules[i], kernel::cpu::name_fft.c_str())) {
                      // found fft kernel in module i
                      which_module[kernel::cpu::name_fft] = i;
                    }
                    if (dlsym(*modules[i], kernel::cpu::name_adder.c_str())) {
                      // found adder kernel in module i
                      which_module[kernel::cpu::name_adder] = i;
                    }
                    if (dlsym(*modules[i], kernel::cpu::name_splitter.c_str())) {
                      // found gridder kernel in module i
                      which_module[kernel::cpu::name_splitter] = i;
                    }
                } // end for
            } // end find_kernel_functions


            // Gridder class
            Gridder::Gridder(runtime::Module &module) :
                _run(module, name_gridder.c_str()) {}

            void Gridder::run(
                    int nr_subgrids, int gridsize, float image_size,
                    float w_offset, int nr_channels, int nr_stations,
                    void *uvw, void *wavenumbers, void *visibilities,
                    void *spheroidal, void *aterm, void *metadata, void *subgrid) {
                  (sig_gridder (void *) _run)(
                  nr_subgrids, gridsize, image_size, w_offset, nr_channels, nr_stations,
                  uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrid);
            }


            // Degridder class
            Degridder::Degridder(runtime::Module &module) :
                _run(module, name_degridder.c_str()) {}

            void Degridder::run(
                    int nr_subgrids, int gridsize, float image_size,
                    float w_offset, int nr_channels, int nr_stations,
                    void *uvw, void *wavenumbers,
                    void *visibilities, void *spheroidal, void *aterm,
                    void *metadata, void *subgrid) {
                  (sig_degridder (void *) _run)(
                  nr_subgrids, gridsize, image_size, w_offset, nr_channels, nr_stations,
                  uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrid);
            }


            // GridFFT class
            GridFFT::GridFFT(runtime::Module &module) :
                _run(module, name_fft.c_str()) {}

            void GridFFT::run(int gridsize, int size, int batch, void *data, int direction) {
                (sig_fft (void *) _run)(gridsize, size, batch, data, direction);
            }


            // Adder class
            Adder::Adder(runtime::Module &module) :
                _run(module, name_adder.c_str()) {}

            void Adder::run(int nr_subgrids, int gridsize, void *metadata, void *subgrid, void *grid) {
                (sig_adder (void *) _run)(nr_subgrids, gridsize, metadata, subgrid, grid);
            }


            // Splitter class
            Splitter::Splitter(runtime::Module &module) :
                _run(module, name_splitter.c_str()) {}

            void Splitter::run(int nr_subgrids, int gridsize, void *metadata, void *subgrid, void *grid) {
                (sig_splitter (void *) _run)(nr_subgrids, gridsize, metadata, subgrid, grid);
            }

        } // namespace cpu
    } // namespace kernel
} // namespace idg
