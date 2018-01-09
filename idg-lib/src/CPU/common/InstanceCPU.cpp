#include <cstdint> // unint64_t
#include <unistd.h> // rmdir

#include <algorithm> // transform

#include "idg-config.h"

#include "InstanceCPU.h"

using namespace std;

namespace idg {
    namespace kernel {
        namespace cpu {

            // Constructor
            InstanceCPU::InstanceCPU(
                string libdir) :
                KernelsInstance(),
                function_gridder(nullptr),
                function_degridder(nullptr),
                function_fft(nullptr),
                function_adder(nullptr),
                function_splitter(nullptr),
                function_adder_wstack(nullptr),
                function_splitter_wstack(nullptr)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                load_shared_objects(libdir);
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
            }

            void InstanceCPU::load_shared_objects(
                    string libdir)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Derive kernel prefix from directory name (Reference -> reference)
                string prefix = libdir;
                std::transform(prefix.begin(), prefix.end(), prefix.begin(), ::tolower);

                // All libraries
                vector<string> lib_names;
                lib_names.push_back("libcpu-" + prefix + "-kernel-gridder.so");
                lib_names.push_back("libcpu-" + prefix + "-kernel-degridder.so");
                lib_names.push_back("libcpu-" + prefix + "-kernel-adder.so");
                lib_names.push_back("libcpu-" + prefix + "-kernel-splitter.so");
                lib_names.push_back("libcpu-" + prefix + "-kernel-fft.so");
                lib_names.push_back("libcpu-" + prefix + "-kernel-adder-wstack.so");
                lib_names.push_back("libcpu-" + prefix + "-kernel-splitter-wstack.so");

                // Get full path to library directory
                string full_libdir = auxiliary::get_lib_dir() + "/idg-cpu/" + libdir;

                for (auto libname : lib_names) {
                    string lib = full_libdir + "/" + libname;

                    #if defined(DEBUG)
                    cout << "Loading: " << lib << endl;
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
            #define sig_splitter_wstack (void (*)(long,long,int,void*,void*,void*))


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
                  nr_subgrids, grid_size, subgrid_size, image_size, w_step, nr_channels, nr_stations,
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
                int grid_size,
                int subgrid_size,
                int nr_w_layers,
                void *metadata,
                void *subgrid,
                void *grid)
            {
                (sig_adder_wstack (void *) *function_adder_wstack)(nr_subgrids, grid_size, subgrid_size, nr_w_layers, metadata, subgrid, grid);
            }

            void InstanceCPU::run_splitter_wstack(
                int nr_subgrids,
                int grid_size,
                int subgrid_size,
                void *metadata,
                void *subgrid,
                void *grid)
            {
                (sig_splitter_wstack (void *) *function_splitter_wstack)(nr_subgrids, grid_size, subgrid_size, metadata, subgrid, grid);
            }

        } // namespace cpu
    } // namespace kernel
} // namespace idg
