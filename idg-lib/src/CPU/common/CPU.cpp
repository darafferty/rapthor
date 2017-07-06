#include <vector>
#include <memory>

#include "fftw3.h"

#include "CPU.h"

using namespace std;
using namespace idg;
using namespace idg::kernel;
using namespace powersensor;

namespace idg {
    namespace proxy {
        namespace cpu {

            // Constructor
            CPU::CPU(
                CompileConstants constants,
                Compiler compiler,
                Compilerflags flags,
                ProxyInfo info) :
                Proxy(constants),
                kernels(constants, compiler, flags, info)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                powerSensor = get_power_sensor(sensor_host);
            }

            // Destructor
            CPU::~CPU()
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Delete power sensor
                delete powerSensor;
            }

            /*
                High level routines
            */
            void CPU::do_gridding(
                const Plan& plan,
                const float w_step,
                const float cell_size,
                const unsigned int kernel_size,
                const Array1D<float>& frequencies,
                const Array3D<Visibility<std::complex<float>>>& visibilities,
                const Array2D<UVWCoordinate<float>>& uvw,
                const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                Grid& grid,
                const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                const Array1D<unsigned int>& aterms_offsets,
                const Array2D<float>& spheroidal)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                Array1D<float> wavenumbers = compute_wavenumbers(frequencies);

                // Proxy constants
                auto subgrid_size     = mConstants.get_subgrid_size();
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
                auto image_size   = cell_size * grid_size;

                try {
                    double runtime = -omp_get_wtime();

                    auto total_nr_subgrids  = plan.get_nr_subgrids();
                    auto total_nr_timesteps = plan.get_nr_timesteps();

                    // Allocate memory for subgrids
                    Array4D<std::complex<float>> subgrids(total_nr_subgrids, nr_polarizations, subgrid_size, subgrid_size);

                    runtime += omp_get_wtime();
                    #if defined (REPORT_TOTAL)
                    auxiliary::report("init", runtime);
                    #endif

                    runtime = -omp_get_wtime();

                    // Run subroutines
                    grid_onto_subgrids(
                        plan,
                        w_step,
                        grid_size,
                        image_size,
                        wavenumbers,
                        visibilities,
                        uvw,
                        spheroidal,
                        aterms,
                        subgrids);

                    add_subgrids_to_grid(
                        plan,
                        w_step,
                        subgrids,
                        grid);

                    runtime += omp_get_wtime();

                    #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                    uint64_t flops_gridder  = kernels.flops_gridder(nr_channels, total_nr_timesteps, total_nr_subgrids);
                    uint64_t bytes_gridder  = kernels.bytes_gridder(nr_channels, total_nr_timesteps, total_nr_subgrids);
                    uint64_t flops_fft      = kernels.flops_fft(subgrid_size, total_nr_subgrids);
                    uint64_t bytes_fft      = kernels.bytes_fft(subgrid_size, total_nr_subgrids);
                    uint64_t flops_adder    = kernels.flops_adder(total_nr_subgrids);
                    uint64_t bytes_adder    = kernels.bytes_adder(total_nr_subgrids);
                    uint64_t flops_gridding = flops_gridder + flops_fft + flops_adder;
                    uint64_t bytes_gridding = bytes_gridder + bytes_fft + bytes_adder;
                    auxiliary::report("|gridding", runtime, flops_gridding, bytes_gridding);

                    auto total_nr_visibilities = plan.get_nr_visibilities();
                    auxiliary::report_visibilities("|gridding", runtime, total_nr_visibilities);
                    clog << endl;
                    #endif

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

            void CPU::do_degridding(
                const Plan& plan,
                const float w_step,
                const float cell_size,
                const unsigned int kernel_size,
                const Array1D<float>& frequencies,
                Array3D<Visibility<std::complex<float>>>& visibilities,
                const Array2D<UVWCoordinate<float>>& uvw,
                const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                const Grid& grid,
                const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                const Array1D<unsigned int>& aterms_offsets,
                const Array2D<float>& spheroidal)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                Array1D<float> wavenumbers = compute_wavenumbers(frequencies);

                // Proxy constants
                auto subgrid_size     = mConstants.get_subgrid_size();
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
                auto image_size   = cell_size * grid_size;

                try {
                    double runtime = -omp_get_wtime();

                    auto total_nr_subgrids  = plan.get_nr_subgrids();
                    auto total_nr_timesteps = plan.get_nr_timesteps();

                    // Allocate memory for subgrids
                    Array4D<std::complex<float>> subgrids(total_nr_subgrids, nr_polarizations, subgrid_size, subgrid_size);

                    runtime += omp_get_wtime();
                    #if defined (REPORT_TOTAL)
                    auxiliary::report("init", runtime);
                    #endif

                    runtime = -omp_get_wtime();

                    // Run subroutines
                    split_grid_into_subgrids(
                         plan,
                         w_step,
                         subgrids,
                         grid);

                    degrid_from_subgrids(
                        plan,
                        w_step,
                        grid_size,
                        image_size,
                        wavenumbers,
                        visibilities,
                        uvw,
                        spheroidal,
                        aterms,
                        subgrids);

                    runtime += omp_get_wtime();

                    #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                    uint64_t flops_degridder  = kernels.flops_degridder(nr_channels, total_nr_timesteps, total_nr_subgrids);
                    uint64_t bytes_degridder  = kernels.bytes_degridder(nr_channels, total_nr_timesteps, total_nr_subgrids);
                    uint64_t flops_fft        = kernels.flops_fft(subgrid_size, total_nr_subgrids);
                    uint64_t bytes_fft        = kernels.bytes_fft(subgrid_size, total_nr_subgrids);
                    uint64_t flops_splitter   = kernels.flops_splitter(total_nr_subgrids);
                    uint64_t bytes_splitter   = kernels.bytes_splitter(total_nr_subgrids);
                    uint64_t flops_degridding = flops_degridder + flops_fft + flops_splitter;
                    uint64_t bytes_degridding = bytes_degridder + bytes_fft + bytes_splitter;
                    auxiliary::report("|degridding", runtime, flops_degridding, bytes_degridding);

                    auto total_nr_visibilities = plan.get_nr_visibilities();
                    auxiliary::report_visibilities("|degridding", runtime, total_nr_visibilities);
                    clog << endl;
                    #endif

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

            void CPU::do_transform(
                DomainAtoDomainB direction,
                Array3D<std::complex<float>>& grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                cout << "FFT (direction: " << direction << ")" << endl;
                #endif

                try {
                    int sign = (direction == FourierDomainToImageDomain) ? 1 : -1;

                    // Constants
                    auto grid_size = grid.get_x_dim();
                    auto nr_correlations = mConstants.get_nr_correlations();

                    // FFT shift
                    if (direction == FourierDomainToImageDomain) {
                        kernels.shift(grid); // TODO: integrate into adder?
                    } else {
                        kernels.shift(grid); // TODO: remove
                    }

                    // Run FFT
                    State powerStates[2];
                    powerStates[0] = powerSensor->read();
                    kernels.run_fft(grid_size, grid_size, 1, grid.data(), sign);
                    powerStates[1] = powerSensor->read();

                    // FFT shift
                    if (direction == FourierDomainToImageDomain)
                        kernels.shift(grid); // TODO: remove
                    else
                        kernels.shift(grid); // TODO: integrate into splitter?

                    // Report performance
                    #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                    auxiliary::report("grid-fft",
                                      kernels.flops_fft(grid_size, 1),
                                      kernels.bytes_fft(grid_size, 1),
                                      powerSensor, powerStates[0], powerStates[1]);
                    clog << endl;
                    #endif
                } catch (const exception& e) {
                    cerr << __func__ << " caught exception: "
                         << e.what() << endl;
                } catch (...) {
                    cerr << __func__ << " caught unknown exception" << endl;
                }
            } // end transform


            /*
                Low level routines
            */
            void CPU::grid_onto_subgrids(
                const Plan& plan,
                const float w_step,
                const unsigned int grid_size,
                const float image_size,
                const Array1D<float>& wavenumbers,
                const Array3D<Visibility<std::complex<float>>>& visibilities,
                const Array2D<UVWCoordinate<float>>& uvw,
                const Array2D<float>& spheroidal,
                const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                Array4D<std::complex<float>>& subgrids)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Constants
                auto nr_baselines = visibilities.get_z_dim();
                auto jobsize      = kernel::cpu::jobsize_gridder;
                auto nr_timesteps = visibilities.get_y_dim();
                auto nr_channels  = visibilities.get_x_dim();
                auto subgrid_size = subgrids.get_y_dim();
                auto nr_stations  = aterms.get_z_dim();

                // Performance measurements
                double total_runtime_gridding = 0;
                double total_runtime_gridder  = 0;
                double total_runtime_fft      = 0;
                State powerStates[4];

                // Start gridder
                for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                    unsigned int first_bl, last_bl, current_nr_baselines;
                    plan.initialize_job(nr_baselines, jobsize, bl, &first_bl, &last_bl, &current_nr_baselines);
                    if (current_nr_baselines == 0) continue;

                    // Initialize iteration
                    auto current_nr_subgrids  = plan.get_nr_subgrids(first_bl, current_nr_baselines);
                    auto current_nr_timesteps = plan.get_nr_timesteps(first_bl, current_nr_baselines);
                    void *wavenumbers_ptr  = wavenumbers.data();
                    void *spheroidal_ptr   = spheroidal.data();
                    void *aterm_ptr        = aterms.data();
                    void *metadata_ptr     = (void *) plan.get_metadata_ptr(first_bl);
                    void *uvw_ptr          = uvw.data(first_bl, 0);
                    void *visibilities_ptr = visibilities.data(first_bl, 0, 0);
                    void *subgrids_ptr     = subgrids.data(plan.get_subgrid_offset(first_bl), 0, 0, 0);

                    // Gridder kernel
                    powerStates[0] = powerSensor->read();

                    kernels.run_gridder(
                        current_nr_subgrids,
                        grid_size,
                        subgrid_size,
                        image_size,
                        w_step,
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
                    kernels.run_fft(grid_size, subgrid_size, current_nr_subgrids, subgrids_ptr, FFTW_BACKWARD);
                    powerStates[3] = powerSensor->read();

                    // Performance reporting
                    #if defined(REPORT_VERBOSE)
                    auxiliary::report("gridder",
                                      kernels.flops_gridder(nr_channels, current_nr_timesteps, current_nr_subgrids),
                                      kernels.bytes_gridder(nr_channels, current_nr_timesteps, current_nr_subgrids),
                                      powerSensor, powerStates[0], powerStates[1]);
                    auxiliary::report("sub-fft",
                                      kernels.flops_fft(subgrid_size, current_nr_subgrids),
                                      kernels.bytes_fft(subgrid_size, current_nr_subgrids),
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
                uint64_t total_flops_gridder  = kernels.flops_gridder(nr_channels, total_nr_timesteps, total_nr_subgrids);
                uint64_t total_bytes_gridder  = kernels.bytes_gridder(nr_channels, total_nr_timesteps, total_nr_subgrids);
                uint64_t total_flops_fft      = kernels.flops_fft(subgrid_size, total_nr_subgrids);
                uint64_t total_bytes_fft      = kernels.bytes_fft(subgrid_size, total_nr_subgrids);
                auxiliary::report("|gridder", total_runtime_gridder, total_flops_gridder, total_bytes_gridder);
                auxiliary::report("|sub-fft", total_runtime_fft, total_flops_fft, total_bytes_fft);
                clog << endl;
                #endif
            } // end grid_onto_subgrids

            void CPU::add_subgrids_to_grid(
                const Plan& plan,
                const float w_step,
                const Array4D<std::complex<float>>& subgrids,
                Grid& grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Constants
                auto jobsize      = kernel::cpu::jobsize_adder;
                auto grid_size    = grid.get_x_dim();
                auto nr_w_layers  = grid.get_w_dim();
                auto nr_baselines = plan.get_nr_baselines();
                auto subgrid_size = subgrids.get_y_dim();

                // Performance measurements
                double total_runtime_adding = 0;
                double total_runtime_adder  = 0;
                total_runtime_adding = -omp_get_wtime();
                State powerStates[2];

                // Run adder
                for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                    // Number of elements in job
                    int current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;
                    auto nr_subgrids = plan.get_nr_subgrids(bl, current_nr_baselines);

                    // Pointers to the first element in processed batch
                    void *metadata_ptr = (void *) plan.get_metadata_ptr(bl);
                    void *subgrids_ptr = subgrids.data(plan.get_subgrid_offset(bl), 0, 0, 0);
                    void *grid_ptr     = grid.data();

                    powerStates[0] = powerSensor->read();
                    if (w_step == 0.0) {
                        kernels.run_adder(nr_subgrids, grid_size, subgrid_size, metadata_ptr, subgrids_ptr, grid_ptr);
                    }
                    else {
                        kernels.run_adder_wstack(nr_subgrids, grid_size, subgrid_size, nr_w_layers, metadata_ptr, subgrids_ptr, grid_ptr);
                    }
                    powerStates[1] = powerSensor->read();

                    #if defined(REPORT_VERBOSE)
                    auxiliary::report("adder",
                                      kernels.flops_adder(nr_subgrids),
                                      kernels.bytes_adder(nr_subgrids),
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
                uint64_t total_flops_adder = kernels.flops_adder(nr_subgrids);
                uint64_t total_bytes_adder = kernels.bytes_adder(nr_subgrids);
                auxiliary::report("|adder", total_runtime_adder, total_flops_adder, total_bytes_adder);
                auxiliary::report("|adding", total_runtime_adding, total_flops_adder, total_bytes_adder);
                auxiliary::report_subgrids("|adding", total_runtime_adding, nr_subgrids);
                clog << endl;
                #endif
            } // end add_subgrids_to_grid

            void CPU::split_grid_into_subgrids(
                const Plan& plan,
                const float w_step,
                Array4D<std::complex<float>>& subgrids,
                const Grid& grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Constants
                auto jobsize      = kernel::cpu::jobsize_splitter;
                auto grid_size    = grid.get_x_dim();
                auto nr_baselines = plan.get_nr_baselines();
                auto subgrid_size = subgrids.get_y_dim();

                // Performance measurements
                double total_runtime_splitting = 0;
                double total_runtime_splitter  = 0;
                total_runtime_splitting = -omp_get_wtime();
                State powerStates[2];

                // Run splitter
                for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                    // Number of elements in job
                    int current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;
                    auto nr_subgrids = plan.get_nr_subgrids(bl, current_nr_baselines);

                    // Pointers to the first element in processed batch
                    void *metadata_ptr = (void *) plan.get_metadata_ptr(bl);
                    void *subgrids_ptr = subgrids.data(plan.get_subgrid_offset(bl), 0, 0, 0);
                    void *grid_ptr     = grid.data();

                    powerStates[0] = powerSensor->read();
                    if (w_step == 0.0) {
                       kernels.run_splitter(nr_subgrids, grid_size, subgrid_size, metadata_ptr, subgrids_ptr, grid_ptr);
                    } else {
                       kernels.run_splitter_wstack(nr_subgrids, grid_size, subgrid_size, metadata_ptr, subgrids_ptr, grid_ptr);
                    }
                    powerStates[1] = powerSensor->read();

                    #if defined(REPORT_VERBOSE)
                    auxiliary::report("splitter",
                                      kernels.flops_splitter(nr_subgrids),
                                      kernels.bytes_splitter(nr_subgrids),
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
                uint64_t total_flops_splitter = kernels.flops_splitter(nr_subgrids);
                uint64_t total_bytes_splitter = kernels.bytes_splitter(nr_subgrids);
                auxiliary::report("|splitter", total_runtime_splitter, total_flops_splitter, total_bytes_splitter);
                auxiliary::report("|splitting", total_runtime_splitting, total_flops_splitter, total_bytes_splitter);
                auxiliary::report_subgrids("|splitting", total_runtime_splitting, nr_subgrids);
                clog << endl;
                #endif
            } // end split_grid_into_subgrids

            void CPU::degrid_from_subgrids(
                const Plan& plan,
                const float w_step,
                const unsigned int grid_size,
                const float image_size,
                const Array1D<float>& wavenumbers,
                Array3D<Visibility<std::complex<float>>>& visibilities,
                const Array2D<UVWCoordinate<float>>& uvw,
                const Array2D<float>& spheroidal,
                const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                const Array4D<std::complex<float>>& subgrids)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Constants
                auto nr_baselines = visibilities.get_z_dim();
                auto jobsize      = kernel::cpu::jobsize_degridder;
                auto nr_timesteps = visibilities.get_y_dim();
                auto nr_channels  = visibilities.get_x_dim();
                auto subgrid_size = subgrids.get_y_dim();
                auto nr_stations  = aterms.get_z_dim();

                // Performance measurements
                double total_runtime_degridding = 0;
                double total_runtime_degridder  = 0;
                double total_runtime_fft        = 0;
                State powerStates[4];

                // Start degridder
                for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                    unsigned int first_bl, last_bl, current_nr_baselines;
                    plan.initialize_job(nr_baselines, jobsize, bl, &first_bl, &last_bl, &current_nr_baselines);
                    if (current_nr_baselines == 0) continue;

                    // Initialize iteration
                    auto current_nr_subgrids  = plan.get_nr_subgrids(first_bl, current_nr_baselines);
                    auto current_nr_timesteps = plan.get_nr_timesteps(first_bl, current_nr_baselines);
                    void *wavenumbers_ptr  = wavenumbers.data();
                    void *spheroidal_ptr   = spheroidal.data();
                    void *aterm_ptr        = aterms.data();
                    void *metadata_ptr     = (void *) plan.get_metadata_ptr(first_bl);
                    void *uvw_ptr          = uvw.data(first_bl, 0);
                    void *visibilities_ptr = visibilities.data(first_bl, 0, 0);
                    void *subgrids_ptr     = subgrids.data(plan.get_subgrid_offset(first_bl), 0, 0, 0);

                    // FFT kernel
                    powerStates[0] = powerSensor->read();
                    kernels.run_fft(grid_size, subgrid_size, current_nr_subgrids, subgrids_ptr, FFTW_FORWARD);
                    powerStates[1] = powerSensor->read();

                    // Degridder kernel
                    powerStates[2] = powerSensor->read();
                    kernels.run_degridder(
                        current_nr_subgrids,
                        grid_size,
                        subgrid_size,
                        image_size,
                        w_step,
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
                                      kernels.flops_degridder(nr_channels, current_nr_timesteps, current_nr_subgrids),
                                      kernels.bytes_degridder(nr_channels, current_nr_timesteps, current_nr_subgrids),
                                      powerSensor, powerStates[2], powerStates[3]);
                    auxiliary::report("sub-fft",
                                      kernels.flops_fft(subgrid_size, current_nr_subgrids),
                                      kernels.flops_fft(subgrid_size, current_nr_subgrids),
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
                uint64_t total_flops_degridder  = kernels.flops_degridder(nr_channels, total_nr_timesteps, total_nr_subgrids);
                uint64_t total_bytes_degridder  = kernels.bytes_degridder(nr_channels, total_nr_timesteps, total_nr_subgrids);
                uint64_t total_flops_fft        = kernels.flops_fft(subgrid_size, total_nr_subgrids);
                uint64_t total_bytes_fft        = kernels.bytes_fft(subgrid_size, total_nr_subgrids);
                uint64_t total_flops_degridding = total_flops_degridder + total_flops_fft;
                uint64_t total_bytes_degridding = total_bytes_degridder + total_bytes_fft;
                auxiliary::report("|degridder", total_runtime_degridder,
                                  total_flops_degridder, total_bytes_degridder);
                auxiliary::report("|sub-fft", total_runtime_fft, total_flops_fft, total_bytes_fft);
                clog << endl;
                #endif
            } // end degrid_from_subgrids


        } // namespace cpu
    } // namespace proxy
} // namespace idg
