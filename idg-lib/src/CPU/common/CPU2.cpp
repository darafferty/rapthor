#include <vector>
#include <memory>

#include "CPU2.h"

using namespace std;
using namespace idg;

namespace idg {
    namespace proxy {
        namespace cpu {

            // Constructor
            CPU2::CPU2(
                CompileConstants constants,
                Compiler compiler,
                Compilerflags flags,
                ProxyInfo info) :
                Proxy2(constants),
                mCompiler(compiler),
                mFlags(flags),
                mInfo(info)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                cout << "Compiler: " << compiler << endl;
                cout << "Compiler flags: " << flags << endl;
                #endif

                compile();
                load_shared_objects();
                find_kernel_functions();

                #if defined(HAVE_LIKWID) && 0
                powerSensor = new LikwidPowerSensor();
                #else
                powerSensor = new RaplPowerSensor();
                #endif
            }

            // Destructor
            CPU2::~CPU2()
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

                // Delete power sensor
                delete powerSensor;
            }

            /*
                High level routines
            */
            void CPU2::gridding(
                const float w_offset,
                const float cell_size,
                const unsigned int kernel_size,
                const Array1D<float>& frequencies,
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

                Array1D<float> wavenumbers = compute_wavenumbers(frequencies);

                // Proxy constants
                auto subgrid_size      = mConstants.get_subgrid_size();
                auto nr_polarizations = mConstants.get_nr_correlations();

                // Checks arguments
                if (kernel_size <= 0 || kernel_size >= subgrid_size-1) {
                    throw invalid_argument("0 < kernel_size < subgrid_size-1 not true");
                }

                check_dimensions(
                    frequencies, visibilities, uvw, baselines,
                    grid, aterms, aterms_offsets, spheroidal);

                // Arguments
                auto nr_baselines = visibilities.get_z_dim();
                auto nr_timesteps = visibilities.get_y_dim();
                auto nr_channels  = visibilities.get_x_dim();
                auto grid_size    = grid.get_x_dim();

                // TODO: remove parameters object from kernels
                Parameters parameters;
                parameters.set_nr_stations(aterms.get_z_dim());
                parameters.set_nr_channels(nr_channels);
                parameters.set_nr_time(nr_timesteps);
                parameters.set_nr_timeslots(aterms.get_w_dim());
                parameters.set_imagesize(cell_size * grid.get_x_dim());
                parameters.set_subgrid_size(subgrid_size);
                parameters.set_grid_size(grid.get_x_dim());
                mParams = parameters;

                try {
                   double runtime = -omp_get_wtime();

                    // Initialize plan
                    Plan2 plan(
                        kernel_size, subgrid_size, grid_size, cell_size,
                        frequencies, uvw, baselines, aterms_offsets);

                    auto total_nr_subgrids  = plan.get_nr_subgrids();
                    auto total_nr_timesteps = plan.get_nr_timesteps();

                    // Allocate memory for subgrids
                    auto size_subgrids = 1ULL * total_nr_subgrids * nr_polarizations *
                                                subgrid_size * subgrid_size;
                    auto subgrids = new complex<float>[size_subgrids];

                    runtime += omp_get_wtime();
                    #if defined (REPORT_TOTAL)
                    auxiliary::report("init", runtime);
                    #endif

                    runtime = -omp_get_wtime();

                    // Run subroutines
                    grid_onto_subgrids(
                        plan,
                        w_offset,
                        (const float*) uvw.data(),
                        (const float*) wavenumbers.data(),
                        (const complex<float>*) visibilities.data(),
                        (const float*) spheroidal.data(),
                        (const complex<float>*) aterms.data(),
                        subgrids);

                    add_subgrids_to_grid(
                        plan,
                        subgrids,
                        (complex<float>*) grid.data());

                    runtime += omp_get_wtime();

                    #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                    unique_ptr<kernel::cpu::Gridder> kernel_gridder = get_kernel_gridder();
                    unique_ptr<kernel::cpu::GridFFT> kernel_fft = get_kernel_fft();
                    unique_ptr<kernel::cpu::Adder> kernel_adder = get_kernel_adder();
                    uint64_t flops_gridder  = kernel_gridder->flops(total_nr_timesteps,
                                                                    total_nr_subgrids);
                    uint64_t bytes_gridder  = kernel_gridder->bytes(total_nr_timesteps,
                                                                    total_nr_subgrids);
                    uint64_t flops_fft      = kernel_fft->flops(subgrid_size,
                                                                total_nr_subgrids);
                    uint64_t bytes_fft      = kernel_fft->bytes(subgrid_size,
                                                                total_nr_subgrids);
                    uint64_t flops_adder    = kernel_adder->flops(total_nr_subgrids);
                    uint64_t bytes_adder    = kernel_adder->bytes(total_nr_subgrids);
                    uint64_t flops_gridding = flops_gridder + flops_fft + flops_adder;
                    uint64_t bytes_gridding = bytes_gridder + bytes_fft + bytes_adder;
                    auxiliary::report("|gridding", runtime,
                        flops_gridding, bytes_gridding);
                    auxiliary::report_visibilities("|gridding",
                        runtime, total_nr_timesteps, nr_channels);
                    clog << endl;
                    #endif

                    delete[] subgrids;

                } catch (const invalid_argument& e) {
                    cerr << __func__ << ": invalid argument: "
                         << e.what() << endl;
                    exit(1);
                } catch (const exception& e) {
                    cerr << __func__ << ": caught exception: "
                         << e.what() << endl;
                    exit(2);
                } catch (...) {
                    cerr << __func__ << ": caught unknown exception" << endl;
                    exit(3);
                }
            } // end gridding

            void CPU2::degridding(
                const float w_offset,
                const float cell_size,
                const unsigned int kernel_size,
                const Array1D<float>& frequencies,
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

                Array1D<float> wavenumbers = compute_wavenumbers(frequencies);

                // Proxy constants
                auto subgrid_size      = mConstants.get_subgrid_size();
                auto nr_polarizations = mConstants.get_nr_correlations();

                // Checks arguments
                if (kernel_size <= 0 || kernel_size >= subgrid_size-1) {
                    throw invalid_argument("0 < kernel_size < subgrid_size-1 not true");
                }

                check_dimensions(
                    frequencies, visibilities, uvw, baselines,
                    grid, aterms, aterms_offsets, spheroidal);

                // Arguments
                auto nr_baselines = visibilities.get_z_dim();
                auto nr_timesteps = visibilities.get_y_dim();
                auto nr_channels  = visibilities.get_x_dim();
                auto grid_size    = grid.get_x_dim();

                // TODO: remove parameters object from kernels
                Parameters parameters;
                parameters.set_nr_stations(aterms.get_z_dim());
                parameters.set_nr_channels(nr_channels);
                parameters.set_nr_time(nr_timesteps);
                parameters.set_nr_timeslots(aterms.get_w_dim());
                parameters.set_imagesize(cell_size * grid.get_x_dim());
                parameters.set_subgrid_size(subgrid_size);
                parameters.set_grid_size(grid.get_x_dim());
                mParams = parameters;

                try {
                    double runtime = -omp_get_wtime();

                    // Initialize plan
                    Plan2 plan(
                        kernel_size, subgrid_size, grid_size, cell_size,
                        frequencies, uvw, baselines, aterms_offsets);

                    auto total_nr_subgrids  = plan.get_nr_subgrids();
                    auto total_nr_timesteps = plan.get_nr_timesteps();

                    // Allocate memory for subgrids
                    auto size_subgrids = 1ULL * total_nr_subgrids * nr_polarizations *
                                                subgrid_size * subgrid_size;
                    auto subgrids = new complex<float>[size_subgrids];

                    runtime += omp_get_wtime();
                    #if defined (REPORT_TOTAL)
                    auxiliary::report("init", runtime);
                    #endif

                    runtime = -omp_get_wtime();

                    // Run subroutines
                    split_grid_into_subgrids(
                         plan,
                         subgrids,
                         (complex<float>*) grid.data());

                    degrid_from_subgrids(
                        plan,
                        w_offset,
                        (const float*) uvw.data(),
                        (const float*) wavenumbers.data(),
                        (complex<float>*) visibilities.data(),
                        (const float*) spheroidal.data(),
                        (const complex<float>*) aterms.data(),
                        (const complex<float>*) subgrids);

                    runtime += omp_get_wtime();

                    #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                    unique_ptr<kernel::cpu::Degridder> kernel_degridder = get_kernel_degridder();
                    unique_ptr<kernel::cpu::GridFFT> kernel_fft = get_kernel_fft();
                    unique_ptr<kernel::cpu::Splitter> kernel_splitter = get_kernel_splitter();
                    uint64_t flops_degridder  = kernel_degridder->flops(total_nr_timesteps,
                                                                        total_nr_subgrids);
                    uint64_t bytes_degridder  = kernel_degridder->bytes(total_nr_timesteps,
                                                                        total_nr_subgrids);
                    uint64_t flops_fft        = kernel_fft->flops(subgrid_size, total_nr_subgrids);
                    uint64_t bytes_fft        = kernel_fft->bytes(subgrid_size, total_nr_subgrids);
                    uint64_t flops_splitter   = kernel_splitter->flops(total_nr_subgrids);
                    uint64_t bytes_splitter   = kernel_splitter->bytes(total_nr_subgrids);
                    uint64_t flops_degridding = flops_degridder + flops_fft + flops_splitter;
                    uint64_t bytes_degridding = bytes_degridder + bytes_fft + bytes_splitter;
                    auxiliary::report("|degridding", runtime,
                        flops_degridding, bytes_degridding);
                    auxiliary::report_visibilities("|degridding",
                        runtime, total_nr_timesteps, nr_channels);
                    clog << endl;
                    #endif

                    delete[] subgrids;

                } catch (const invalid_argument& e) {
                    cerr << __func__ << ": invalid argument: "
                         << e.what() << endl;
                    exit(1);
                } catch (const exception& e) {
                    cerr << __func__ << ": caught exception: "
                         << e.what() << endl;
                    exit(2);
                } catch (...) {
                    cerr << __func__ << ": caught unknown exception" << endl;
                    exit(3);
                }
            } // end degridding

            void CPU2::transform(
                DomainAtoDomainB direction,
                const Array3D<std::complex<float>>& grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
            }

            /*
                Low level routines
            */
            void CPU2::grid_onto_subgrids(
                const Plan2& plan,
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

                // Constants
                auto jobsize          = mParams.get_job_size_gridder();
                auto nr_baselines     = mParams.get_nr_baselines();
                auto nr_time          = mParams.get_nr_time();
                auto nr_channels      = mParams.get_nr_channels();
                auto nr_stations      = mParams.get_nr_stations();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgrid_size     = mParams.get_subgrid_size();
                auto gridsize         = mParams.get_grid_size();
                auto imagesize        = mParams.get_imagesize();

                // Load kernel functions
                unique_ptr<kernel::cpu::Gridder> kernel_gridder = get_kernel_gridder();
                unique_ptr<kernel::cpu::GridFFT> kernel_fft = get_kernel_fft();

                // Performance measurements
                double total_runtime_gridding = 0;
                double total_runtime_gridder  = 0;
                double total_runtime_fft      = 0;
                PowerSensor::State powerStates[4];

                // Start gridder
                for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                    // Number of elements per baseline
                    auto uvw_elements          = nr_time * sizeof(UVW)/sizeof(float);
                    auto visibilities_elements = nr_time * nr_channels * nr_polarizations;

                    // Number of baselines in job
                    int current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;

                    // Number of subgrids for all baselines in job
                    auto current_nr_subgrids   = plan.get_nr_subgrids(bl, current_nr_baselines);
                    auto current_nr_timesteps  = plan.get_nr_timesteps(bl, current_nr_baselines);
                    auto subgrid_elements      = subgrid_size * subgrid_size * nr_polarizations;

                    // Pointers to the first element in processed batch
                    void *wavenumbers_ptr  = const_cast<float*>(wavenumbers);
                    void *spheroidal_ptr   = const_cast<float*>(spheroidal);
                    void *aterm_ptr        = const_cast<complex<float>*>(aterm);
                    void *uvw_ptr          = const_cast<float*>(uvw + bl * uvw_elements);
                    void *visibilities_ptr = const_cast<complex<float>*>(visibilities
                                             + bl * visibilities_elements);
                    void *metadata_ptr     = (void *) plan.get_metadata_ptr(bl);
                    void *subgrids_ptr     = subgrids + subgrid_elements * plan.get_subgrid_offset(bl);

                    // Gridder kernel
                    powerStates[0] = powerSensor->read();

                    kernel_gridder->run(
                        current_nr_subgrids,
                        gridsize,
                        imagesize,
                        w_offset,
                        nr_channels,
                        nr_stations,
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
                    kernel_fft->run(gridsize, subgrid_size, current_nr_subgrids, subgrids_ptr, FFTW_BACKWARD);
                    powerStates[3] = powerSensor->read();

                    // Performance reporting
                    #if defined(REPORT_VERBOSE)
                    auxiliary::report("gridder",
                                      kernel_gridder->flops(current_nr_timesteps, current_nr_subgrids),
                                      kernel_gridder->bytes(current_nr_timesteps, current_nr_subgrids),
                                      powerSensor, powerStates[0], powerStates[1]);
                    auxiliary::report("sub-fft",
                                      kernel_fft->flops(subgrid_size, current_nr_subgrids),
                                      kernel_fft->bytes(subgrid_size, current_nr_subgrids),
                                      powerSensor, powerStates[2], powerStates[3]);
                    #endif
                    #if defined(REPORT_TOTAL)
                    double runtime_gridder = powerSensor->seconds(powerStates[0], powerStates[1]);
                    double runtime_fft     = powerSensor->seconds(powerStates[2], powerStates[3]);
                    total_runtime_gridder += runtime_gridder;
                    total_runtime_fft     += runtime_fft;
                    #endif
                } // end for bl

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                clog << endl;
                auto total_nr_subgrids  = plan.get_nr_subgrids();
                auto total_nr_timesteps = plan.get_nr_timesteps();
                uint64_t total_flops_gridder  = kernel_gridder->flops(total_nr_timesteps,
                                                                      total_nr_subgrids);
                uint64_t total_bytes_gridder  = kernel_gridder->bytes(total_nr_timesteps,
                                                                      total_nr_subgrids);
                uint64_t total_flops_fft      = kernel_fft->flops(subgrid_size, total_nr_subgrids);
                uint64_t total_bytes_fft      = kernel_fft->bytes(subgrid_size, total_nr_subgrids);
                auxiliary::report("|gridder", total_runtime_gridder,
                                  total_flops_gridder, total_bytes_gridder);
                auxiliary::report("|sub-fft", total_runtime_fft, total_flops_fft,
                                  total_bytes_fft);
                clog << endl;
                #endif
            } // end grid_onto_subgrids

            void CPU2::add_subgrids_to_grid(
                const Plan2& plan,
                const complex<float> *subgrids,
                complex<float> *grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Constants
                auto jobsize          = mParams.get_job_size_adder();
                auto nr_baselines     = mParams.get_nr_baselines();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgrid_size     = mParams.get_subgrid_size();
                auto gridsize         = mParams.get_grid_size();

                // Load kernel function
                unique_ptr<kernel::cpu::Adder> kernel_adder = get_kernel_adder();

                // Performance measurements
                double total_runtime_adding = 0;
                double total_runtime_adder  = 0;
                total_runtime_adding = -omp_get_wtime();
                PowerSensor::State powerStates[2];

                // Run adder
                for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                    // Number of baselines in job
                    int current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;

                    // Number of elements in batch
                    auto nr_subgrids = plan.get_nr_subgrids(bl, current_nr_baselines);
                    auto elems_per_subgrid = subgrid_size * subgrid_size *
                                             nr_polarizations;

                    // Pointers to the first element in processed batch
                    void *subgrid_ptr  = const_cast<complex<float>*>(subgrids
                                         + elems_per_subgrid*plan.get_subgrid_offset(bl));
                    void *grid_ptr     = grid;
                    void *metadata_ptr = (void *) plan.get_metadata_ptr(bl);

                    powerStates[0] = powerSensor->read();
                    kernel_adder->run(nr_subgrids, gridsize, metadata_ptr, subgrid_ptr, grid_ptr);
                    powerStates[1] = powerSensor->read();

                    #if defined(REPORT_VERBOSE)
                    auxiliary::report("adder",
                                      kernel_adder->flops(nr_subgrids),
                                      kernel_adder->bytes(nr_subgrids),
                                      powerSensor, powerStates[0], powerStates[1]);
                    #endif
                    #if defined(REPORT_TOTAL)
                    double runtime_adder = powerSensor->seconds(powerStates[0], powerStates[1]);
                    total_runtime_adder += runtime_adder;
                    #endif
                } // end for bl

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                total_runtime_adding += omp_get_wtime();
                clog << endl;
                auto nr_subgrids = plan.get_nr_subgrids();
                uint64_t total_flops_adder = kernel_adder->flops(nr_subgrids);
                uint64_t total_bytes_adder = kernel_adder->bytes(nr_subgrids);
                auxiliary::report("|adder", total_runtime_adder, total_flops_adder, total_bytes_adder);
                auxiliary::report("|adding", total_runtime_adding, total_flops_adder, total_bytes_adder);
                auxiliary::report_subgrids("|adding", total_runtime_adding, nr_subgrids);
                clog << endl;
                #endif
            } // end add_subgrids_to_grid

            void CPU2::split_grid_into_subgrids(
                const Plan2& plan,
                complex<float> *subgrids,
                const complex<float> *grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Constants
                auto jobsize          = mParams.get_job_size_splitter();
                auto nr_baselines     = mParams.get_nr_baselines();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgrid_size     = mParams.get_subgrid_size();
                auto gridsize         = mParams.get_grid_size();

                // Load kernel function
                unique_ptr<kernel::cpu::Splitter> kernel_splitter = get_kernel_splitter();

                // Performance measurements
                double total_runtime_splitting = 0;
                double total_runtime_splitter = 0;
                total_runtime_splitting = -omp_get_wtime();
                PowerSensor::State powerStates[2];

                // Run splitter
                for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                    // Number of baselines in job
                    int current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;

                    // Number of elements in batch
                    auto nr_subgrids = plan.get_nr_subgrids(bl, current_nr_baselines);
                    auto elems_per_subgrid = subgrid_size * subgrid_size
                                             * nr_polarizations;

                    // Pointers to the first element in processed batch
                    void *subgrid_ptr  = subgrids
                                         + elems_per_subgrid*plan.get_subgrid_offset(bl);
                    void *grid_ptr     = const_cast<complex<float>*>(grid);
                    void *metadata_ptr = (void *) plan.get_metadata_ptr(bl);

                    powerStates[0] = powerSensor->read();
                    kernel_splitter->run(nr_subgrids, gridsize, metadata_ptr, subgrid_ptr, grid_ptr);
                    powerStates[1] = powerSensor->read();

                    #if defined(REPORT_VERBOSE)
                    auxiliary::report("splitter",
                                      kernel_splitter->flops(nr_subgrids),
                                      kernel_splitter->bytes(nr_subgrids),
                                      powerSensor, powerStates[0], powerStates[1]);
                    #endif
                    #if defined(REPORT_TOTAL)
                    double runtime_splitter = powerSensor->seconds(powerStates[0], powerStates[1]);
                    total_runtime_splitter += runtime_splitter;
                    #endif
                } // end for bl

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                total_runtime_splitting += omp_get_wtime();
                clog << endl;
                auto nr_subgrids = plan.get_nr_subgrids();
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
            } // end split_grid_into_subgrids

            void CPU2::degrid_from_subgrids(
                const Plan2& plan,
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

                // Constants
                auto jobsize          = mParams.get_job_size_gridder();
                auto nr_baselines     = mParams.get_nr_baselines();
                auto nr_time          = mParams.get_nr_time();
                auto nr_channels      = mParams.get_nr_channels();
                auto nr_stations      = mParams.get_nr_stations();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgrid_size     = mParams.get_subgrid_size();
                auto gridsize         = mParams.get_grid_size();
                auto imagesize        = mParams.get_imagesize();

                // Load kernel functions
                unique_ptr<kernel::cpu::Degridder> kernel_degridder = get_kernel_degridder();
                unique_ptr<kernel::cpu::GridFFT> kernel_fft = get_kernel_fft();

                // Performance measurements
                double total_runtime_degridding = 0;
                double total_runtime_degridder = 0;
                double total_runtime_fft = 0;
                PowerSensor::State powerStates[4];

                // Start degridder
                for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                    // Number of elements per baseline
                    auto uvw_elements          = nr_time * sizeof(UVW)/sizeof(float);
                    auto visibilities_elements = nr_time * nr_channels * nr_polarizations;

                    // Number of baselines in job
                    int current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;

                    // Number of subgrids for all baselines in job
                    auto current_nr_subgrids  = plan.get_nr_subgrids(bl, current_nr_baselines);
                    auto current_nr_timesteps = plan.get_nr_timesteps(bl, current_nr_baselines);
                    auto subgrid_elements     = subgrid_size * subgrid_size * nr_polarizations;

                    // Pointers to the first element in processed batch
                    void *wavenumbers_ptr  = const_cast<float*>(wavenumbers);
                    void *spheroidal_ptr   = const_cast<float*>(spheroidal);
                    void *aterm_ptr        = const_cast<complex<float>*>(aterm);
                    void *uvw_ptr          = const_cast<float*>(uvw + bl * uvw_elements);
                    void *visibilities_ptr = visibilities
                                             + bl * visibilities_elements;
                    void *metadata_ptr     = (void *) plan.get_metadata_ptr(bl);
                    void *subgrids_ptr     = const_cast<complex<float>*>(subgrids
                                             + subgrid_elements * plan.get_subgrid_offset(bl));

                    // FFT kernel
                    powerStates[0] = powerSensor->read();
                    kernel_fft->run(gridsize, subgrid_size, current_nr_subgrids, subgrids_ptr, FFTW_FORWARD);
                    powerStates[1] = powerSensor->read();

                    // Degridder kernel
                    powerStates[2] = powerSensor->read();
                    kernel_degridder->run(
                        current_nr_subgrids,
                        gridsize,
                        imagesize,
                        w_offset,
                        nr_channels,
                        nr_stations,
                        uvw_ptr,
                        wavenumbers_ptr,
                        visibilities_ptr,
                        spheroidal_ptr,
                        aterm_ptr,
                        metadata_ptr,
                        subgrids_ptr);
                    powerStates[3] = powerSensor->read();

                    // Performance reporting
                    #if defined(REPORT_VERBOSE)
                    auxiliary::report("degridder",
                                      kernel_degridder->flops(current_nr_timesteps, current_nr_subgrids),
                                      kernel_degridder->bytes(current_nr_timesteps, current_nr_subgrids),
                                      powerSensor, powerStates[2], powerStates[3]);
                    auxiliary::report("sub-fft",
                                      kernel_fft->flops(subgrid_size, current_nr_subgrids),
                                      kernel_fft->bytes(subgrid_size, current_nr_subgrids),
                                      powerSensor, powerStates[0], powerStates[1]);
                    #endif
                    #if defined(REPORT_TOTAL)
                    double runtime_fft       = powerSensor->seconds(powerStates[0], powerStates[1]);
                    double runtime_degridder = powerSensor->seconds(powerStates[2], powerStates[3]);
                    total_runtime_fft       += runtime_fft;
                    total_runtime_degridder += runtime_degridder;
                    #endif
                } // end for bl

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                clog << endl;
                auto total_nr_subgrids  = plan.get_nr_subgrids();
                auto total_nr_timesteps = plan.get_nr_timesteps();
                uint64_t total_flops_degridder  = kernel_degridder->flops(total_nr_timesteps,
                                                                          total_nr_subgrids);
                uint64_t total_bytes_degridder  = kernel_degridder->bytes(total_nr_timesteps,
                                                                          total_nr_subgrids);
                uint64_t total_flops_fft        = kernel_fft->flops(subgrid_size,
                                                                    total_nr_subgrids);
                uint64_t total_bytes_fft        = kernel_fft->bytes(subgrid_size,
                                                                    total_nr_subgrids);
                uint64_t total_flops_degridding = total_flops_degridder + total_flops_fft;
                uint64_t total_bytes_degridding = total_bytes_degridder + total_bytes_fft;
                auxiliary::report("|degridder", total_runtime_degridder,
                                  total_flops_degridder, total_bytes_degridder);
                auxiliary::report("|sub-fft", total_runtime_fft, total_flops_fft, total_bytes_fft);
                clog << endl;
                #endif
            } // end degrid_from_subgrids


            /*
                 Runtime compilation
            */
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

            void CPU2::compile()
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Set compile arguments:
                stringstream arguments;
                arguments << "-DNR_POLARIZATIONS=" << mConstants.get_nr_correlations();
                arguments << " -DSUBGRIDSIZE=" << mConstants.get_subgrid_size();
                arguments << " " << mFlags;
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
                        mCompiler.c_str(),
                        lib.c_str(),
                        arguments.str().c_str());
                } // end for each library
            } // end compile

            void CPU2::load_shared_objects()
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
            void CPU2::find_kernel_functions()
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

            unique_ptr<kernel::cpu::Gridder> CPU2::get_kernel_gridder() const {
                return unique_ptr<kernel::cpu::Gridder>(
                    new kernel::cpu::Gridder(
                        *(modules[which_module.at(kernel::cpu::name_gridder)]),
                        mParams));
            }


            unique_ptr<kernel::cpu::Degridder> CPU2::get_kernel_degridder() const {
                return unique_ptr<kernel::cpu::Degridder>(
                    new kernel::cpu::Degridder(
                        *(modules[which_module.at(kernel::cpu::name_degridder)]),
                        mParams));
            }


            unique_ptr<kernel::cpu::Adder> CPU2::get_kernel_adder() const {
                return unique_ptr<kernel::cpu::Adder>(
                    new kernel::cpu::Adder(
                        *(modules[which_module.at(kernel::cpu::name_adder)]),
                        mParams));
            }

            unique_ptr<kernel::cpu::Splitter> CPU2::get_kernel_splitter() const {
                return unique_ptr<kernel::cpu::Splitter>(
                    new kernel::cpu::Splitter(
                        *(modules[which_module.at(kernel::cpu::name_splitter)]),
                        mParams));
            }


            unique_ptr<kernel::cpu::GridFFT> CPU2::get_kernel_fft() const {
                return unique_ptr<kernel::cpu::GridFFT>(
                    new kernel::cpu::GridFFT(
                        *(modules[which_module.at(kernel::cpu::name_fft)]),
                        mParams));
            }

        } // namespace cpu
    } // namespace proxy
} // namespace idg
