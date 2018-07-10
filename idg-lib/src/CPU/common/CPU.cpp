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
                std::vector<std::string> libraries):
                kernels(libraries)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                powerSensor = get_power_sensor(sensor_host);
                kernels.set_report(report);
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
                const unsigned int subgrid_size,
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

                // Checks arguments
                if (kernel_size <= 0 || kernel_size >= subgrid_size-1) {
                    throw invalid_argument("0 < kernel_size < subgrid_size-1 not true");
                }

                // Arguments
                auto nr_baselines = visibilities.get_z_dim();
                auto nr_timesteps = visibilities.get_y_dim();
                auto nr_channels  = visibilities.get_x_dim();
                auto grid_size    = grid.get_x_dim();
                auto image_size   = cell_size * grid_size;

                try {
                    auto total_nr_subgrids  = plan.get_nr_subgrids();
                    auto total_nr_timesteps = plan.get_nr_timesteps();

                    // Allocate memory for subgrids
                    Array4D<std::complex<float>> subgrids(total_nr_subgrids, nr_polarizations, subgrid_size, subgrid_size);

                    // Performance measurement
                    report.initialize(nr_channels, subgrid_size, grid_size);
                    State states[2];
                    states[0] = powerSensor->read();

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

                    states[1] = powerSensor->read();
                    report.update_host(states[0], states[1]);

                    // Performance report
                    #if defined(REPORT_TOTAL)
                    report.print_total(total_nr_timesteps, total_nr_subgrids);
                    auto total_nr_visibilities = plan.get_nr_visibilities();
                    report.print_visibilities(auxiliary::name_gridding, total_nr_visibilities);
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
                const unsigned int subgrid_size,
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

                // Checks arguments
                if (kernel_size <= 0 || kernel_size >= subgrid_size-1) {
                    throw invalid_argument("0 < kernel_size < subgrid_size-1 not true");
                }

                // Arguments
                auto nr_baselines = visibilities.get_z_dim();
                auto nr_timesteps = visibilities.get_y_dim();
                auto nr_channels  = visibilities.get_x_dim();
                auto grid_size    = grid.get_x_dim();
                auto image_size   = cell_size * grid_size;

                try {
                    auto total_nr_subgrids  = plan.get_nr_subgrids();
                    auto total_nr_timesteps = plan.get_nr_timesteps();

                    // Allocate memory for subgrids
                    Array4D<std::complex<float>> subgrids(total_nr_subgrids, nr_polarizations, subgrid_size, subgrid_size);

                    // Performance measurement
                    report.initialize(nr_channels, subgrid_size, grid_size);
                    State states[2];
                    states[0] = powerSensor->read();

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

                    states[1] = powerSensor->read();
                    report.update_host(states[0], states[1]);

                    // Report performance
                    #if defined(REPORT_TOTAL)
                    report.print_total(total_nr_timesteps, total_nr_subgrids);
                    auto total_nr_visibilities = plan.get_nr_visibilities();
                    report.print_visibilities(auxiliary::name_degridding, total_nr_visibilities);
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

                    // Performance measurement
                    report.initialize(0, 0, grid_size);
                    kernels.set_report(report);
                    State states[2];
                    states[0] = powerSensor->read();

                    // FFT shift
                    if (direction == FourierDomainToImageDomain) {
                        kernels.shift(grid); // TODO: integrate into adder?
                    } else {
                        kernels.shift(grid); // TODO: remove
                    }

                    // Run FFT
                    kernels.run_fft(grid_size, grid_size, 1, grid.data(), sign);

                    // FFT shift
                    if (direction == FourierDomainToImageDomain)
                        kernels.shift(grid); // TODO: remove
                    else
                        kernels.shift(grid); // TODO: integrate into splitter?

                    // End measurement
                    states[1] = powerSensor->read();
                    report.update_host(states[0], states[1]);

                    // Report performance
                    #if defined(REPORT_TOTAL)
                    report.print_total();
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

                    // FFT kernel
                    kernels.run_subgrid_fft(grid_size, subgrid_size, current_nr_subgrids, subgrids_ptr, FFTW_BACKWARD);

                    // Performance reporting
                    #if defined(REPORT_VERBOSE)
                    report.print(current_nr_timesteps, current_nr_subgrids);
                    #endif
                } // end for bl
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

                // Run adder
                for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                    // Number of elements in job
                    int current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;
                    auto current_nr_subgrids = plan.get_nr_subgrids(bl, current_nr_baselines);

                    // Pointers to the first element in processed batch
                    void *metadata_ptr = (void *) plan.get_metadata_ptr(bl);
                    void *subgrids_ptr = subgrids.data(plan.get_subgrid_offset(bl), 0, 0, 0);
                    void *grid_ptr     = grid.data();

                    if (w_step == 0.0) {
                        kernels.run_adder(current_nr_subgrids, grid_size, subgrid_size, metadata_ptr, subgrids_ptr, grid_ptr);
                    }
                    else {
                        kernels.run_adder_wstack(current_nr_subgrids, grid_size, subgrid_size, nr_w_layers, metadata_ptr, subgrids_ptr, grid_ptr);
                    }

                    #if defined(REPORT_VERBOSE)
                    report.print(0, current_nr_subgrids);
                    #endif
                } // end for bl
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

                // Run splitter
                for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                    // Number of elements in job
                    int current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;
                    auto current_nr_subgrids = plan.get_nr_subgrids(bl, current_nr_baselines);

                    // Pointers to the first element in processed batch
                    void *metadata_ptr = (void *) plan.get_metadata_ptr(bl);
                    void *subgrids_ptr = subgrids.data(plan.get_subgrid_offset(bl), 0, 0, 0);
                    void *grid_ptr     = grid.data();

                    if (w_step == 0.0) {
                       kernels.run_splitter(current_nr_subgrids, grid_size, subgrid_size, metadata_ptr, subgrids_ptr, grid_ptr);
                    } else {
                       kernels.run_splitter_wstack(current_nr_subgrids, grid_size, subgrid_size, metadata_ptr, subgrids_ptr, grid_ptr);
                    }

                    #if defined(REPORT_VERBOSE) | defined(REPORT_TOTAL)
                    report.print(0, current_nr_subgrids);
                    #endif
                } // end for bl
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
                    kernels.run_subgrid_fft(grid_size, subgrid_size, current_nr_subgrids, subgrids_ptr, FFTW_FORWARD);

                    // Degridder kernel
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

                    // Performance reporting
                    #if defined(REPORT_VERBOSE)
                    report.print(current_nr_timesteps, current_nr_subgrids);
                    #endif
                } // end for bl
            } // end degrid_from_subgrids


        } // namespace cpu
    } // namespace proxy
} // namespace idg
