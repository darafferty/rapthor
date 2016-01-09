#include <cstdio> // remove()
#include <cstdlib>  // rand()
#include <ctime> // time() to init srand()
#include <complex>
#include <sstream>
#include <memory>
#include <dlfcn.h> // dlsym()
#include <omp.h> // omp_get_wtime
#include <libgen.h> // dirname() and basename()
#include <unistd.h> // rmdir()

#include "idg-config.h"
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

                powerSensor = new LikwidPowerSensor();
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

                try {

                    // initialize metadata
                    vector<Metadata> metadata = init_metadata(
                        uvw, wavenumbers, baselines, aterm_offsets, kernel_size);
                    auto nr_subgrids = metadata.size();

                    // allocate 'subgrids' memory for subgrids
                    auto nr_baselines = mParams.get_nr_baselines();
                    auto nr_timeslots = mParams.get_nr_timeslots();
                    auto nr_polarizations = mParams.get_nr_polarizations();;
                    auto subgridsize = mParams.get_subgrid_size();
                    auto size_subgrids = 1ULL*nr_subgrids*nr_polarizations*
                                         subgridsize*subgridsize;
                    auto subgrids = new complex<float>[size_subgrids];

                    grid_onto_subgrids(
                        nr_subgrids,
                        w_offset,
                        uvw,
                        wavenumbers,
                        visibilities,
                        spheroidal,
                        aterm,
                        metadata,
                        subgrids);

                    add_subgrids_to_grid(
                        nr_subgrids,
                        metadata,
                        subgrids,
                        grid);

                    delete[] subgrids;

                } catch (const exception& e) {
                    cerr << __func__ << " caught exception: "
                         << e.what() << endl;
                } catch (...) {
                    cerr << __func__ << " caught unknown exception" << endl;
                }
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

                try {
                    // initialize metadata
                    vector<Metadata> metadata = init_metadata(
                        uvw, wavenumbers, baselines, aterm_offsets, kernel_size);
                    auto nr_subgrids = metadata.size();

                    // allocate 'subgrids' memory for subgrids
                    auto nr_baselines = mParams.get_nr_baselines();
                    auto nr_timeslots = mParams.get_nr_timeslots();
                    auto nr_polarizations = mParams.get_nr_polarizations();;
                    auto subgridsize = mParams.get_subgrid_size();
                    auto size_subgrids = 1ULL * nr_subgrids*nr_polarizations*
                                         subgridsize*subgridsize;
                    auto subgrids = new complex<float>[size_subgrids];

                    split_grid_into_subgrids(
                         nr_subgrids,
                         metadata,
                         subgrids,
                         grid);

                    degrid_from_subgrids(
                        nr_subgrids,
                        w_offset,
                        uvw,
                        wavenumbers,
                        visibilities,
                        spheroidal,
                        aterm,
                        metadata,
                        subgrids);

                    delete[] subgrids;

                } catch (const exception& e) {
                    cerr << __func__ << " caught exception: "
                         << e.what() << endl;
                } catch (...) {
                    cerr << __func__ << " caught unknown exception" << endl;
                }
            }


            void CPU::transform(DomainAtoDomainB direction,
                                complex<float>* grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                cout << "Transform direction: " << direction << endl;
                #endif

                try {

                    int sign = (direction == FourierDomainToImageDomain) ? 1 : -1;

                    // Constants
                    auto gridsize = mParams.get_grid_size();

                    // Load kernel function
                    unique_ptr<kernel::cpu::GridFFT> kernel_fft = get_kernel_fft();

                    // Start fft
                    double runtime = -omp_get_wtime();
                    kernel_fft->run(gridsize, 1, grid, sign);
                    runtime += omp_get_wtime();

                    #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                    auxiliary::report(">grid_fft", runtime,
                        kernel_fft->flops(gridsize, 1),
                        kernel_fft->bytes(gridsize, 1));
                    clog << endl;
                    #endif

                } catch (const exception& e) {
                    cerr << __func__ << " caught exception: "
                         << e.what() << endl;
                } catch (...) {
                    cerr << __func__ << " caught unknown exception" << endl;
                }
            }


            /*
                Low level routines
            */
            void CPU::grid_onto_subgrids(
                const unsigned nr_subgrids,
                const float w_offset,
                const float *uvw,
                const float *wavenumbers,
                const complex<float> *visibilities,
                const float *spheroidal,
                const complex<float> *aterm,
                const vector<Metadata>& metadata,
                complex<float> *subgrids)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Constants
                auto jobsize = mParams.get_job_size_gridder();
                auto nr_baselines = mParams.get_nr_baselines();
                auto nr_timesteps = mParams.get_nr_timesteps();
                auto nr_timeslots = mParams.get_nr_timeslots();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();

                // load kernel functions
                unique_ptr<kernel::cpu::Gridder> kernel_gridder = get_kernel_gridder();
                unique_ptr<kernel::cpu::GridFFT> kernel_fft = get_kernel_fft();

                // Performance measurements
                double total_runtime_gridding = 0;
                double total_runtime_gridder = 0;
                double total_runtime_fft = 0;
                LikwidPowerSensor::State powerStates[4];
                total_runtime_gridding -= omp_get_wtime();

                // Start gridder
                // TODO: loop over baseline?
                for (unsigned int s = 0; s < nr_subgrids; s += jobsize) {

                    // Prevent overflow
                    jobsize = s + jobsize > nr_subgrids ? nr_subgrids - s : jobsize;

                    // offset for arrays with [baseline*nr_time + time]
                    int offset       = metadata[s].offset;

                    int uvw_elements = sizeof(UVW)/sizeof(float); // = 3
                    int visibilities_elements = nr_channels * nr_polarizations;
                    int subgrid_elements      = subgridsize * subgridsize
                                                 * nr_polarizations;
                    // TODO: if later metadata back to int*:
                    // int metadata_elements = sizeof(Metadata)/sizeof(int);

                    // Note: the following are the pointers to the first
                    // element that is processed in the kernel
                    // These are mainly determined as in other code,
                    // where the kernels are offload these parts
                    // need to be copied over
                    void *wavenumbers_ptr  = const_cast<float*>(wavenumbers);
                    void *spheroidal_ptr   = const_cast<float*>(spheroidal);
                    void *aterm_ptr        = const_cast<complex<float>*>(aterm);
                    void *uvw_ptr          = (float *) uvw + offset * uvw_elements;
                    void *visibilities_ptr = (complex<float>*) visibilities
                                             + offset * visibilities_elements;
                    void *subgrids_ptr     = (complex<float>*) subgrids
                                             + s * subgrid_elements;
                    void *metadata_ptr     = (void *) &(metadata[s]);
                    // TODO: if later metadata back to int*:
                    // void *metadata_ptr     = (int *) metadata + s * metadata_elements;

                    // Gridder kernel
                    powerStates[0] = powerSensor->read();

                    kernel_gridder->run(
                        jobsize,
                        w_offset,
                        uvw_ptr,
                        wavenumbers_ptr,
                        visibilities_ptr,
                        spheroidal_ptr,
                        aterm_ptr,
                        metadata_ptr,
                        subgrids_ptr
                        );

                    powerStates[1] = powerSensor->read();

                    // FFT kernel
                    powerStates[2] = powerSensor->read();
                    kernel_fft->run(subgridsize, jobsize, subgrids_ptr, FFTW_BACKWARD);
                    powerStates[3] = powerSensor->read();

                    // Performance reporting
                    double runtime_gridder = LikwidPowerSensor::seconds(
                        powerStates[0], powerStates[1] );
                    double runtime_fft     = LikwidPowerSensor::seconds(
                        powerStates[2], powerStates[3] );
                    #if defined(REPORT_VERBOSE)
                    double power_gridder   = LikwidPowerSensor::Watt(
                        powerStates[0], powerStates[1] );
                    double power_fft       = LikwidPowerSensor::Watt(
                        powerStates[2], powerStates[3] );
                    auxiliary::report("gridder", runtime_gridder,
                                      kernel_gridder->flops(jobsize),
                                      kernel_gridder->bytes(jobsize),
                                      power_gridder);
                    auxiliary::report("fft", runtime_fft,
                                      kernel_fft->flops(subgridsize, jobsize),
                                      kernel_fft->bytes(subgridsize, jobsize),
                                      power_fft);
                    #endif
                    #if defined(REPORT_TOTAL)
                    total_runtime_gridder += runtime_gridder;
                    total_runtime_fft += runtime_fft;
                    #endif
                } // end for s

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                total_runtime_gridding += omp_get_wtime();
                clog << endl;
                uint64_t total_flops_gridder  = kernel_gridder->flops(nr_subgrids);
                uint64_t total_bytes_gridder  = kernel_gridder->bytes(nr_subgrids);
                uint64_t total_flops_fft      = kernel_fft->flops(subgridsize,
                                                                 nr_subgrids);
                uint64_t total_bytes_fft      = kernel_fft->bytes(subgridsize,
                                                                 nr_subgrids);
                uint64_t total_flops_gridding = total_flops_gridder + total_flops_fft;
                uint64_t total_bytes_gridding = total_bytes_gridder + total_bytes_fft;
                auxiliary::report("|gridder", total_runtime_gridder,
                                  total_flops_gridder, total_bytes_gridder);
                auxiliary::report("|fft", total_runtime_fft, total_flops_fft,
                                  total_bytes_fft);
                auxiliary::report("|gridding", total_runtime_gridding,
                                  total_flops_gridding, total_bytes_gridding);
                auxiliary::report_visibilities("|gridding", total_runtime_gridding,
                                               nr_baselines,
                                               nr_timesteps * nr_timeslots,
                                               nr_channels);
                clog << endl;
                #endif
            }


            void CPU::add_subgrids_to_grid(
                const unsigned nr_subgrids,
                const vector<Metadata>& metadata,
                const complex<float> *subgrids,
                complex<float> *grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Constants
                auto jobsize = mParams.get_job_size_adder();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();

                // Load kernel function
                unique_ptr<kernel::cpu::Adder> kernel_adder = get_kernel_adder();

                // Performance measurements
                double total_runtime_adding = 0;
                double total_runtime_adder = 0;
                total_runtime_adding = -omp_get_wtime();

                // Run adder
                for (unsigned int s = 0; s < nr_subgrids; s += jobsize) {
                    // Prevent overflow
                    jobsize = s + jobsize > nr_subgrids ? nr_subgrids - s: jobsize;

                    // Number of elements in batch

                    // TODO: if later metadata back to int*:
                    // int metadata_elements = sizeof(Metadata)/sizeof(int);;
                    int subgrid_elements  = subgridsize * subgridsize
                                            * nr_polarizations;

                    // Pointer to data for current jobs
                    // TODO: if later metadata back to int*:
                    // void *metadata_ptr = (int *) metadata + s * metadata_elements;
                    void *metadata_ptr = (int *) &(metadata[s]);
                    void *subgrid_ptr  = (complex<float>*) subgrids
                                         + s * subgrid_elements;
                    void *grid_ptr     = grid;

                    double runtime_adder = -omp_get_wtime();
                    kernel_adder->run(jobsize, metadata_ptr, subgrid_ptr, grid_ptr);
                    runtime_adder += omp_get_wtime();

                    #if defined(REPORT_VERBOSE)
                    auxiliary::report("adder", runtime_adder,
                                      kernel_adder->flops(jobsize),
                                      kernel_adder->bytes(jobsize));
                    #endif
                    #if defined(REPORT_TOTAL)
                    total_runtime_adder += runtime_adder;
                    #endif
                } // end for s

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                total_runtime_adding += omp_get_wtime();
                clog << endl;
                uint64_t total_flops_adder = kernel_adder->flops(nr_subgrids);
                uint64_t total_bytes_adder = kernel_adder->bytes(nr_subgrids);
                auxiliary::report("|adder", total_runtime_adder, total_flops_adder,
                                  total_bytes_adder);
                auxiliary::report("|adding", total_runtime_adding, total_flops_adder,
                                  total_bytes_adder);
                auxiliary::report_subgrids("|adding", total_runtime_adding,
                                           nr_subgrids);
                clog << endl;
                #endif
            }


            void CPU::split_grid_into_subgrids(
                const unsigned nr_subgrids,
                const vector<Metadata>& metadata,
                complex<float> *subgrids,
                const complex<float> *grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Constants
                auto jobsize = mParams.get_job_size_splitter();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();

                // Load kernel function
                unique_ptr<kernel::cpu::Splitter> kernel_splitter = get_kernel_splitter();

                // Performance measurements
                double total_runtime_splitting = 0;
                double total_runtime_splitter = 0;
                total_runtime_splitting = -omp_get_wtime();

                // Run splitter
                for (unsigned int s = 0; s < nr_subgrids; s += jobsize) {
                    // Prevent overflow
                    jobsize = s + jobsize > nr_subgrids ? nr_subgrids - s : jobsize;

                    // Number of elements in batch
                    // TODO: if later metadata back to int*:
                    // int metadata_elements = sizeof(Metadata)/sizeof(int);;
                    int subgrid_elements  = subgridsize * subgridsize
                                            * nr_polarizations;

                    // Pointer to data for current jobs
                    void *subgrid_ptr  = (complex<float>*) subgrids
                                         + s * subgrid_elements;
                    void *grid_ptr     = const_cast<complex<float>*>(grid);
                    void *metadata_ptr = (int *) &(metadata[s]);
                    // TODO: if later metadata back to int*:
                    // void *metadata_ptr = (int *) metadata + s * metadata_elements;

                    double runtime_splitter = -omp_get_wtime();
                    kernel_splitter->run(jobsize, metadata_ptr, subgrid_ptr, grid_ptr);
                    runtime_splitter += omp_get_wtime();

                    #if defined(REPORT_VERBOSE)
                    auxiliary::report("splitter", runtime_splitter,
                                      kernel_splitter->flops(jobsize),
                                      kernel_splitter->bytes(jobsize));
                    #endif
                    #if defined(REPORT_TOTAL)
                    total_runtime_splitter += runtime_splitter;
                    #endif
                } // end for bl

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                total_runtime_splitting += omp_get_wtime();
                clog << endl;
                uint64_t total_flops_splitter = kernel_splitter->flops(nr_subgrids);
                uint64_t total_bytes_splitter = kernel_splitter->bytes(nr_subgrids);
                auxiliary::report("|splitter", total_runtime_splitter,
                                  total_flops_splitter, total_bytes_splitter);
                auxiliary::report("|splitting", total_runtime_splitting,
                                  total_flops_splitter, total_bytes_splitter);
                auxiliary::report_subgrids("|splitting", total_runtime_splitting,
                                           nr_subgrids);
                clog << endl;
                #endif
            }


            void CPU::degrid_from_subgrids(
                const unsigned nr_subgrids,
                const float w_offset,
                const float *uvw,
                const float *wavenumbers,
                std::complex<float> *visibilities,
                const float *spheroidal,
                const std::complex<float> *aterm,
                const vector<Metadata>& metadata,
                const std::complex<float> *subgrids)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Constants
                auto jobsize = mParams.get_job_size_degridder();
                auto nr_baselines = mParams.get_nr_baselines();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_timesteps = mParams.get_nr_timesteps();
                auto nr_timeslots = mParams.get_nr_timeslots();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();

                // Load kernel functions
                unique_ptr<kernel::cpu::Degridder> kernel_degridder = get_kernel_degridder();
                unique_ptr<kernel::cpu::GridFFT> kernel_fft = get_kernel_fft();

                // Performance measurements
                double total_runtime_degridding = 0;
                double total_runtime_degridder = 0;
                double total_runtime_fft = 0;
                LikwidPowerSensor::State powerStates[4];
                total_runtime_degridding = -omp_get_wtime();

                // Start degridder
                for (unsigned int s = 0; s < nr_subgrids; s += jobsize) {
                    // Prevent overflow
                    jobsize = s + jobsize > nr_subgrids ? nr_subgrids - s : jobsize;

                    // offset for arrays with [baseline*nr_time + time]
                    int offset       = metadata[s].offset;

                    int uvw_elements = sizeof(UVW)/sizeof(float); // = 3
                    int visibilities_elements = nr_channels * nr_polarizations;
                    int subgrid_elements      = subgridsize * subgridsize
                                                 * nr_polarizations;
                    // TODO: if later metadata back to int*:
                    // int metadata_elements = sizeof(Metadata)/sizeof(int);

                    // Note: the following are the pointers to the first
                    // element that is processed in the kernel
                    // These are mainly determined as in other code,
                    // where the kernels are offload these parts
                    // need to be copied over
                    void *wavenumbers_ptr  = const_cast<float*>(wavenumbers);
                    void *spheroidal_ptr   = const_cast<float*>(spheroidal);
                    void *aterm_ptr        = const_cast<complex<float>*>(aterm);
                    void *uvw_ptr          = (float *) uvw + offset * uvw_elements;
                    void *visibilities_ptr = (complex<float>*) visibilities
                                             + offset * visibilities_elements;
                    void *subgrids_ptr     = (complex<float>*) subgrids
                                             + s * subgrid_elements;
                    void *metadata_ptr     = (void *) &(metadata[s]);
                    // TODO: if later metadata back to int*:
                    // void *metadata_ptr     = (int *) metadata + s * metadata_elements;

                    // FFT kernel
                    powerStates[0] = powerSensor->read();
                    kernel_fft->run(subgridsize, jobsize, subgrids_ptr, FFTW_FORWARD);
                    powerStates[1] = powerSensor->read();

                    // Degridder kernel
                    powerStates[2] = powerSensor->read();
                    kernel_degridder->run(
                        jobsize,
                        w_offset,
                        uvw_ptr,
                        wavenumbers_ptr,
                        visibilities_ptr,
                        spheroidal_ptr,
                        aterm_ptr,
                        metadata_ptr,
                        subgrids_ptr);
                    powerStates[3] = powerSensor->read();

                    // Performance reporting
                    double runtime_fft         = LikwidPowerSensor::seconds(
                        powerStates[0], powerStates[1] );
                    double runtime_degridder   = LikwidPowerSensor::seconds(
                        powerStates[2], powerStates[3] );
                    #if defined(REPORT_VERBOSE)
                    double power_fft           = LikwidPowerSensor::Watt(
                        powerStates[0], powerStates[1] );
                    double power_degridder     = LikwidPowerSensor::Watt(
                        powerStates[2], powerStates[3] );

                    auxiliary::report("degridder", runtime_degridder,
                                      kernel_degridder->flops(jobsize),
                                      kernel_degridder->bytes(jobsize),
                                      power_degridder);
                    auxiliary::report("fft", runtime_fft,
                                      kernel_fft->flops(subgridsize, jobsize),
                                      kernel_fft->bytes(subgridsize, jobsize),
                                      power_fft);
                    #endif
                    #if defined(REPORT_TOTAL)
                    total_runtime_fft += runtime_fft;
                    total_runtime_degridder += runtime_degridder;
                    #endif
                } // end for s

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                total_runtime_degridding += omp_get_wtime();
                clog << endl;
                uint64_t total_flops_degridder  = kernel_degridder->flops(nr_subgrids);
                uint64_t total_bytes_degridder  = kernel_degridder->bytes(nr_subgrids);
                uint64_t total_flops_fft        = kernel_fft->flops(subgridsize,
                                                                   nr_subgrids);
                uint64_t total_bytes_fft        = kernel_fft->bytes(subgridsize,
                                                                   nr_subgrids);
                uint64_t total_flops_degridding = total_flops_degridder
                                                  + total_flops_fft;
                uint64_t total_bytes_degridding = total_bytes_degridder
                                                  + total_bytes_fft;
                auxiliary::report("|degridder", total_runtime_degridder,
                                  total_flops_degridder, total_bytes_degridder);
                auxiliary::report("|fft", total_runtime_fft, total_flops_fft,
                                  total_bytes_fft);
                auxiliary::report("|degridding", total_runtime_degridding,
                                  total_flops_degridding, total_bytes_degridding);
                auxiliary::report_visibilities("|degridding", total_runtime_degridding,
                                               nr_baselines,
                                               nr_timesteps * nr_timeslots,
                                               nr_channels);
                clog << endl;
                #endif
            }


            void CPU::compile(Compiler compiler, Compilerflags flags)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Set compile options: -DNR_STATIONS=... -DNR_BASELINES=... [...]
                string mparameters =  Parameters::definitions(
                  mParams.get_nr_stations(),
                  mParams.get_nr_baselines(),
                  mParams.get_nr_channels(),
                  mParams.get_nr_timesteps(),
                  mParams.get_nr_timeslots(),
                  mParams.get_imagesize(),
                  mParams.get_nr_polarizations(),
                  mParams.get_grid_size(),
                  mParams.get_subgrid_size());

                string compiler_parameters;
                #if defined(USING_GNU_CXX_COMPILER)
                compiler_parameters = "-DUSING_GNU_CXX_COMPILER";
                #elif defined(USING_INTEL_CXX_COMPILER)
                compiler_parameters = "-DUSING_INTEL_CXX_COMPILER";
                #endif

                string parameters = " " + flags + " " + mparameters +
                                    " " + compiler_parameters;

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
