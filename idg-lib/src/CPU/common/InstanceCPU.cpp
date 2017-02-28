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
                mInfo(info),
                function_adder_wstack(nullptr),
                function_splitter_wstack(nullptr)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                cout << "Compiler: " << compiler << endl;
                cout << "Compiler flags: " << flags << endl;
                #endif

                compile(compiler, flags);
                load_shared_objects();
                load_kernel_funcions();
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

                // Unload functions
                delete function_gridder;
                delete function_degridder;
                delete function_fft;
                delete function_adder;
                delete function_splitter;
                delete function_adder_wstack;
                delete function_splitter_wstack;

                // Delete .so files
                if (mInfo.delete_shared_objects()) {
                    for (auto libname : mInfo.get_lib_names()) {
                        string lib = mInfo.get_path_to_lib() + "/" + libname;
                        remove(lib.c_str());
                    }
                    rmdir(mInfo.get_path_to_lib().c_str());
                }
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
                p.add_src_file_to_lib(libadder, "KernelAdderWStack.cpp", true);
                p.add_src_file_to_lib(libsplitter, "KernelSplitter.cpp");
                p.add_src_file_to_lib(libsplitter, "KernelSplitterWStack.cpp", true);

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
                arguments << " " << flags;

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
            void InstanceCPU::load_kernel_funcions()
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                for (unsigned int i=0; i<modules.size(); i++) {
                    if (dlsym(*modules[i], kernel::cpu::name_gridder.c_str())) {
                        function_gridder = new runtime::Function(*modules[i], name_gridder.c_str());
                    }
                    if (dlsym(*modules[i], kernel::cpu::name_degridder.c_str())) {
                        function_degridder = new runtime::Function(*modules[i], name_degridder.c_str());
                    }
                    if (dlsym(*modules[i], kernel::cpu::name_fft.c_str())) {
                        function_fft = new runtime::Function(*modules[i], name_fft.c_str());
                    }
                    if (dlsym(*modules[i], kernel::cpu::name_adder.c_str())) {
                        function_adder = new runtime::Function(*modules[i], name_adder.c_str());
                    }
                    if (dlsym(*modules[i], kernel::cpu::name_splitter.c_str())) {
                        function_splitter = new runtime::Function(*modules[i], name_splitter.c_str());
                    }
                    if (dlsym(*modules[i], kernel::cpu::name_adder_wstack.c_str())) {
                        function_adder_wstack = new runtime::Function(*modules[i], name_adder_wstack.c_str());
                    }
                    if (dlsym(*modules[i], kernel::cpu::name_splitter_wstack.c_str())) {
                        function_splitter_wstack = new runtime::Function(*modules[i], name_splitter_wstack.c_str());
                    }
                } // end for
            } // end load_kernel_funcions


            // Function signatures
            #define sig_gridder         (void (*)(int,int,int,float,float,int,int,void*,void*,void*,void*,void*,void*,void*))
            #define sig_degridder       (void (*)(int,int,int,float,float,int,int,void*,void*,void*,void*,void*,void*,void*))
            #define sig_fft		        (void (*)(long,long,long,void*,int))
            #define sig_adder	        (void (*)(long,long,int,void*,void*,void*))
            #define sig_splitter        (void (*)(long,long,int,void*,void*,void*))
            #define sig_adder_wstack    (void (*)(long,long,int,int,void*,void*,void*))
            #define sig_splitter_wstack (void (*)(long,long,int,int,void*,void*,void*))


            void InstanceCPU::run_gridder(
                int nr_subgrids,
                int grid_size,
                int subgrid_size,
                float image_size,
                float w_step,
                int nr_channels,
                int nr_stations,
                void *uvw,
                void *wavenumbers,
                void *visibilities,
                void *spheroidal,
                void *aterm,
                void *metadata,
                void *subgrid)
            {
                  (sig_gridder (void *) *function_gridder)(
                  nr_subgrids, gridsize, image_size, w_step, nr_channels, nr_stations,
                  uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrid);
            }

            void InstanceCPU::run_degridder(
                int nr_subgrids,
                int grid_size,
                int subgrid_size,
                float image_size,
                float w_step,
                int nr_channels,
                int nr_stations,
                void *uvw,
                void *wavenumbers,
                void *visibilities,
                void *spheroidal,
                void *aterm,
                void *metadata,
                void *subgrid)
            {
                  (sig_degridder (void *) *function_degridder)(
                  nr_subgrids, grid_size, subgrid_size, image_size, w_step, nr_channels, nr_stations,
                  uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrid);
            }

            void InstanceCPU::run_fft(
                int grid_size,
                int size,
                int batch,
                void *data,
                int direction)
            {
                (sig_fft (void *) *function_fft)(grid_size, size, batch, data, direction);
            }

            void InstanceCPU::run_adder(
                int nr_subgrids,
                int grid_size,
                int subgrid_size,
                void *metadata,
                void *subgrid,
                void *grid)
            {
                (sig_adder (void *) *function_adder)(nr_subgrids, grid_size, subgrid_size, metadata, subgrid, grid);
            }

            void InstanceCPU::run_splitter(
                int nr_subgrids,
                int grid_size,
                int subgrid_size,
                void *metadata,
                void *subgrid,
                void *grid)
            {
                (sig_splitter (void *) *function_splitter)(nr_subgrids, grid_size, subgrid_size, metadata, subgrid, grid);
            }

            void InstanceCPU::run_adder_wstack(
                int nr_subgrids,
                int gridsize,
                int nr_w_layers,
                void *metadata,
                void *subgrid,
                void *grid)
            {
                (sig_adder_wstack (void *) *function_adder_wstack)(nr_subgrids, gridsize, nr_w_layers, metadata, subgrid, grid);
            }

            void InstanceCPU::run_splitter_wstack(
                int nr_subgrids,
                int gridsize,
                int nr_w_layers,
                void *metadata,
                void *subgrid,
                void *grid)
            {
                (sig_splitter_wstack (void *) *function_splitter_wstack)(nr_subgrids, gridsize, nr_w_layers, metadata, subgrid, grid);
            }

        } // namespace cpu
    } // namespace kernel
} // namespace idg
