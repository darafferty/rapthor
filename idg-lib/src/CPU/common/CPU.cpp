#include "CPU.h"

using namespace std;
using namespace idg;

namespace idg {
    namespace proxy {
        namespace cpu {

            /// Constructors
            CPU::CPU(
                Parameters params,
                Compiler compiler,
                Compilerflags flags,
                ProxyInfo info)
              : mInfo(info)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                cout << "Compiler: " << compiler << endl;
                cout << "Compiler flags: " << flags << endl;
                cout << params;
                #endif

                mParams = params;
                parameter_sanity_check(); // throws exception if bad parameters
                compile(compiler, flags);
                load_shared_objects();
                find_kernel_functions();

                #if defined(HAVE_LIKWID) && 0
                powerSensor = new LikwidPowerSensor();
                #else
                powerSensor = new RaplPowerSensor();
                #endif

                // Setup benchmark
                init_benchmark();
            }


            CPU::~CPU()
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // unload shared objects by ~Module
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


            string CPU::make_tempdir()
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


            ProxyInfo CPU::default_proxyinfo(string srcdir, string tmpdir)
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

            void CPU::init_benchmark() {
                char *char_nr_repetitions = getenv("NR_REPETITIONS");
                if (char_nr_repetitions) {
                    nr_repetitions = atoi(char_nr_repetitions);
                    enable_benchmark = nr_repetitions > 1;
                }
                if (enable_benchmark) {
                    std::clog << "Benchmark mode enabled, nr_repetitions = " << nr_repetitions << std::endl;
                }
            }


            /* High level routines */
            void CPU::grid_visibilities(
                const complex<float> *visibilities,
                const float *uvw,
                const float *wavenumbers,
                const int *baselines,
                complex<float> *grid,
                const float w_offset,
                const int kernel_size,
                const complex<float> *aterm,
                const int *aterm_offsets,
                const float *spheroidal)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

            }


            void CPU::degrid_visibilities(
                std::complex<float> *visibilities,
                const float *uvw,
                const float *wavenumbers,
                const int *baselines,
                const std::complex<float> *grid,
                const float w_offset,
                const int kernel_size,
                const std::complex<float> *aterm,
                const int *aterm_offsets,
                const float *spheroidal)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
            }


            void CPU::transform(DomainAtoDomainB direction,
                                complex<float>* grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                cout << "FFT (direction: " << direction << ")" << endl;
                #endif
            }


            /*
                Low level routines
            */
            void CPU::grid_onto_subgrids(
                const Plan& plan,
                const float w_offset,
                const float *uvw,
                const float *wavenumbers,
                const complex<float> *visibilities,
                const float *spheroidal,
                const complex<float> *aterm,
                complex<float> *subgrids)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
            }


            void CPU::add_subgrids_to_grid(
                const Plan& plan,
                const complex<float> *subgrids,
                complex<float> *grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
            }


            void CPU::split_grid_into_subgrids(
                const Plan& plan,
                complex<float> *subgrids,
                const complex<float> *grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
            }


            void CPU::degrid_from_subgrids(
                const Plan& plan,
                const float w_offset,
                const float *uvw,
                const float *wavenumbers,
                std::complex<float> *visibilities,
                const float *spheroidal,
                const std::complex<float> *aterm,
                const std::complex<float> *subgrids)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
            }


            void CPU::ifftshift(int nr_polarizations, complex<float> *grid)
            {
                #if defined(DEBUG)
                cout << __func__ << " (calling fftshift)" << endl;
                #endif

                // TODO: implement for odd size gridsize
                // For even gridsize, same as fftshift
                fftshift(nr_polarizations, grid);
            }


            void CPU::fftshift(int nr_polarizations, complex<float> *grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Note: grid[NR_POLARIZATIONS][GRIDSIZE][GRIDSIZE]
                auto gridsize = mParams.get_grid_size();

                #pragma omp parallel for
                for (int p = 0; p < nr_polarizations; p++) {
                    // Pass &grid[p][GRIDSIZE][GRIDSIZE]
                    // and do shift for each polarization
                    fftshift(grid + p*gridsize*gridsize);
                }
            }


            void CPU::ifftshift(complex<float> *array)
            {
                // TOD: implement
            }


            void CPU::fftshift(complex<float> *array)
            {
                auto gridsize = mParams.get_grid_size();
                auto buffer   = new complex<float>[gridsize];

                if (gridsize % 2 != 0)
                    throw invalid_argument("gridsize is assumed to be even");

                for (int i = 0; i < gridsize/2; i++) {
                    // save i-th row into buffer
                    memcpy(buffer, &array[i*gridsize],
                           gridsize*sizeof(complex<float>));

                    auto j = i + gridsize/2;
                    memcpy(&array[i*gridsize + gridsize/2], &array[j*gridsize],
                           (gridsize/2)*sizeof(complex<float>));
                    memcpy(&array[i*gridsize], &array[j*gridsize + gridsize/2],
                           (gridsize/2)*sizeof(complex<float>));
                    memcpy(&array[j*gridsize], &buffer[gridsize/2],
                           (gridsize/2)*sizeof(complex<float>));
                    memcpy(&array[j*gridsize + gridsize/2], &buffer[0],
                           (gridsize/2)*sizeof(complex<float>));
                }

                delete [] buffer;
            }


            void CPU::compile(Compiler compiler, Compilerflags flags)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Set compile options: -DNR_STATIONS=... -DNR_BASELINES=... [...]
                string mparameters =  Parameters::definitions(
                  mParams.get_nr_polarizations(),
                  mParams.get_subgrid_size());

                stringstream parameters_;
                parameters_ << " " << flags;
                parameters_ << " " << mparameters;
                #if defined(HAVE_MKL)
                parameters_ << " -DHAVE_MKL";
                parameters_ << " -I" << MKL_INCLUDE_DIRS;
                #endif
                string parameters = parameters_.str();

                vector<string> v = mInfo.get_lib_names();

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

                    runtime::Source(source.str().c_str()).compile(compiler.c_str(),
                                                            lib.c_str(),
                                                            parameters.c_str());
                } // for each library
            } // compile

            void CPU::parameter_sanity_check()
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
            }


            void CPU::load_shared_objects()
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
            }


            /// maps name -> index in modules that contain that symbol
            void CPU::find_kernel_functions()
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


            unique_ptr<kernel::cpu::Gridder> CPU::get_kernel_gridder() const {
                return unique_ptr<kernel::cpu::Gridder>(
                    new kernel::cpu::Gridder(
                        *(modules[which_module.at(kernel::cpu::name_gridder)]),
                        mParams));
            }


            unique_ptr<kernel::cpu::Degridder> CPU::get_kernel_degridder() const {
                return unique_ptr<kernel::cpu::Degridder>(
                    new kernel::cpu::Degridder(
                        *(modules[which_module.at(kernel::cpu::name_degridder)]),
                        mParams));
            }


            unique_ptr<kernel::cpu::Adder> CPU::get_kernel_adder() const {
                return unique_ptr<kernel::cpu::Adder>(
                    new kernel::cpu::Adder(
                        *(modules[which_module.at(kernel::cpu::name_adder)]),
                        mParams));
            }

            unique_ptr<kernel::cpu::Splitter> CPU::get_kernel_splitter() const {
                return unique_ptr<kernel::cpu::Splitter>(
                    new kernel::cpu::Splitter(
                        *(modules[which_module.at(kernel::cpu::name_splitter)]),
                        mParams));
            }


            unique_ptr<kernel::cpu::GridFFT> CPU::get_kernel_fft() const {
                return unique_ptr<kernel::cpu::GridFFT>(
                    new kernel::cpu::GridFFT(
                        *(modules[which_module.at(kernel::cpu::name_fft)]),
                        mParams));
            }

        } // namespace cpu
    } // namespace proxy
} // namespace idg


// C interface:
// Rationale: calling the code from C code and Fortran easier,
// and bases to create interface to scripting languages such as
// Python, Julia, Matlab, ...
extern "C" {
    typedef idg::proxy::cpu::CPU CPU;

    void CPU_set_job_size_gridder(CPU* p, int n) {
        p->set_job_size_gridder(n); }
    void CPU_set_job_size_adder(CPU* p, int n) {
        p->set_job_size_adder(n); }
    void CPU_set_job_size_splitter(CPU* p, int n) {
        p->set_job_size_splitter(n); }
    void CPU_set_job_size_degridder(CPU* p, int n) {
        p->set_job_size_degridder(n); }
}  // end extern "C"
