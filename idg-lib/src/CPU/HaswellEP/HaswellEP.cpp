// TODO: check which include files are really necessary
#include <cstdio> // remove()
#include <cstdlib>  // rand()
#include <ctime> // time() to init srand()
#include <complex>
#include <sstream>
#include <memory>
#include <dlfcn.h> // dlsym()
#include <omp.h> // omp_get_wtime
#include <libgen.h> // dirname() and basename()

#include "idg-config.h"
#include "HaswellEP.h"
#if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
#include "auxiliary.h"
#endif

using namespace std;

void dummy() {};

namespace idg {

    namespace proxy {

        /// Constructors
        HaswellEP::HaswellEP(
            Compiler compiler,
            Compilerflags flags,
            Parameters params,
            ProxyInfo info)
            : CPU(compiler, flags, params, info)
        {
            #if defined(DEBUG)
            cout << "HaswellEP::" << __func__ << endl;
            cout << "Compiler: " << compiler << endl;
            cout << "Compiler flags: " << flags << endl;
            cout << params;
            #endif
        }


        // HaswellEP::HaswellEP(
        //     CompilerEnvironment cc,
        //     Parameters params,
        //     ProxyInfo info)
        //     : CPU(compiler, flags, params, info)
        // {
        //     #if defined(DEBUG)
        //     cout << "HaswellEP::" << __func__ << endl;
        //     #endif

        //     // find out which compiler to use
        //     // call HaswellEP(compiler, flags, params, algparams)
        //     cerr << "Constructor not implemented yet" << endl;
        // }


        // HaswellEP::~HaswellEP()
        // {
        //     #if defined(DEBUG)
        //     cout << "HaswellEP::" << __func__ << endl;
        //     #endif

        //     // cerr << "Destructor not implemented yet" << endl;
        //     // // unload shared objects by ~Module
        //     // for (unsigned int i=0; i<modules.size(); i++) {
        //     //     delete modules[i];
        //     // }

        //     // // Delete .so files
        //     // if( mInfo.delete_shared_objects() ) {
        //     //     for (auto libname : mInfo.get_lib_names()) {
        //     //         string lib = mInfo.get_path_to_lib() + "/" + libname;
        //     //         remove(lib.c_str());
        //     //     }
        //     // }
        // }


        ProxyInfo HaswellEP::default_info()
        {
            #if defined(DEBUG)
            cout << "HaswellEP::" << __func__ << endl;
            #endif

            // Find library path
            Dl_info dl_info;
            dladdr((void *) dummy, &dl_info);

            // Derive name of library and location
            string libdir = dirname((char *) dl_info.dli_fname);
            string bname = basename((char *) dl_info.dli_fname);
            cout << "Module " << bname << " loaded from: " 
                 << libdir << endl;

            string  srcdir = string(IDG_SOURCE_DIR) 
                + "/src/CPU/HaswellEP/kernels";

            #if defined(DEBUG)
            cout << "Searching for source files in: " << srcdir << endl;
            #endif
 
            // Create temp directory
            char _tmpdir[] = "/tmp/idg-XXXXXX";
            char *tmpdir = mkdtemp(_tmpdir);
            #if defined(DEBUG)
            cout << "Temporary files will be stored in: " << tmpdir << endl;
            #endif
 
            // Create proxy info
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

        
        string HaswellEP::default_compiler() 
        {
            // TODO: return different ones like in CPU.cpp
            // I guess, this is better than forcing to use one compiler
            return "icpc";
        }
        

        string HaswellEP::default_compiler_flags() 
        {
            // TODO: return different ones like in CPU.cpp
            // I guess, this is better than forcing to use one compiler
            return "-Wall -O3 -fopenmp -mkl -lmkl_avx2 -lmkl_vml_avx2";
        }

    } // namespace proxy

} // namespace idg
