#include <vector>
#include <memory>

#include "fftw3.h"

#include "CPU.h"

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
                std::cout << __func__ << std::endl;
                #endif

                powerSensor = get_power_sensor(sensor_host);
                kernels.set_report(report);
            }

            // Destructor
            CPU::~CPU()
            {
                #if defined(DEBUG)
                std::cout << __func__ << std::endl;
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
                const Array1D<float>& shift,
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
                std::cout << __func__ << std::endl;
                #endif

                Array1D<float> wavenumbers = compute_wavenumbers(frequencies);

                // Checks arguments
                if (kernel_size <= 0 || kernel_size >= subgrid_size-1) {
                    throw std::invalid_argument("0 < kernel_size < subgrid_size-1 not true");
                }

                // Arguments
                auto nr_baselines = visibilities.get_z_dim();
                auto nr_channels  = visibilities.get_x_dim();
                auto grid_size    = grid.get_x_dim();
                auto image_size   = cell_size * grid_size;
                auto nr_stations  = aterms.get_z_dim();

                try {
                    auto jobsize = kernel::cpu::jobsize_gridding;

                    // Allocate memory for subgrids
                    int max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);
                    Array4D<std::complex<float>> subgrids(max_nr_subgrids, nr_polarizations, subgrid_size, subgrid_size);

                    // Performance measurement
                    report.initialize(nr_channels, subgrid_size, grid_size);
                    State states[2];
                    states[0] = powerSensor->read();

                    // Start gridder
                    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                        unsigned int first_bl, last_bl, current_nr_baselines;
                        plan.initialize_job(nr_baselines, jobsize, bl, &first_bl, &last_bl, &current_nr_baselines);
                        if (current_nr_baselines == 0) continue;

                        // Initialize iteration
                        auto current_nr_subgrids  = plan.get_nr_subgrids(first_bl, current_nr_baselines);
                        const float *shift_ptr = shift.data();
                        void *wavenumbers_ptr  = wavenumbers.data();
                        void *spheroidal_ptr   = spheroidal.data();
                        void *aterm_ptr        = aterms.data();
                        void *avg_aterm_ptr    = m_avg_aterm_correction.size() ? m_avg_aterm_correction.data() : nullptr;
                        void *metadata_ptr     = (void *) plan.get_metadata_ptr(first_bl);
                        void *uvw_ptr          = uvw.data(first_bl, 0);
                        void *visibilities_ptr = visibilities.data(first_bl, 0, 0);
                        void *subgrids_ptr     = subgrids.data(0, 0, 0, 0);
                        void *grid_ptr         = grid.data();

                        // Gridder kernel
                        kernels.run_gridder(
                            current_nr_subgrids, grid_size, subgrid_size, image_size, w_step, shift_ptr, nr_channels, nr_stations,
                            uvw_ptr, wavenumbers_ptr, visibilities_ptr, spheroidal_ptr, aterm_ptr, avg_aterm_ptr,
                            metadata_ptr, subgrids_ptr);

                        // FFT kernel
                        kernels.run_subgrid_fft(grid_size, subgrid_size, current_nr_subgrids, subgrids_ptr, FFTW_BACKWARD);

                        // Adder kernel
                        if (w_step == 0.0) {
                            kernels.run_adder(
                                current_nr_subgrids, grid_size, subgrid_size,
                                metadata_ptr, subgrids_ptr, grid_ptr);
                        } else {
                            kernels.run_adder_wstack(
                                current_nr_subgrids, grid_size, subgrid_size,
                                metadata_ptr, subgrids_ptr, grid_ptr);
                        }

                        // Performance reporting
                        #if defined(REPORT_VERBOSE)
                        auto current_nr_timesteps = plan.get_nr_timesteps(first_bl, current_nr_baselines);
                        report.print(current_nr_timesteps, current_nr_subgrids);
                        #endif
                    } // end for bl

                    states[1] = powerSensor->read();
                    report.update_host(states[0], states[1]);

                    // Performance report
                    #if defined(REPORT_TOTAL)
                    auto total_nr_subgrids  = plan.get_nr_subgrids();
                    auto total_nr_timesteps = plan.get_nr_timesteps();
                    report.print_total(total_nr_timesteps, total_nr_subgrids);
                    auto total_nr_visibilities = plan.get_nr_visibilities();
                    report.print_visibilities(auxiliary::name_gridding, total_nr_visibilities);
                    #endif

                } catch (const std::invalid_argument& e) {
                    std::cerr << __func__ << ": invalid argument: "
                         << e.what() << std::endl;
                    exit(1);
                } catch (const std::exception& e) {
                    std::cerr << __func__ << ": caught exception: "
                         << e.what() << std::endl;
                    exit(2);
                } catch (...) {
                    std::cerr << __func__ << ": caught unknown exception" << std::endl;
                    exit(3);
                }
            } // end gridding

            void CPU::do_degridding(
                const Plan& plan,
                const float w_step,
                const Array1D<float>& shift,
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
                std::cout << __func__ << std::endl;
                #endif

                Array1D<float> wavenumbers = compute_wavenumbers(frequencies);

                // Checks arguments
                if (kernel_size <= 0 || kernel_size >= subgrid_size-1) {
                    throw std::invalid_argument("0 < kernel_size < subgrid_size-1 not true");
                }

                // Arguments
                auto nr_baselines = visibilities.get_z_dim();
                auto nr_channels  = visibilities.get_x_dim();
                auto grid_size    = grid.get_x_dim();
                auto image_size   = cell_size * grid_size;
                auto nr_stations  = aterms.get_z_dim();

                try {
                    auto jobsize = kernel::cpu::jobsize_degridding;

                    // Allocate memory for subgrids
                    int max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);
                    Array4D<std::complex<float>> subgrids(max_nr_subgrids, nr_polarizations, subgrid_size, subgrid_size);

                    // Performance measurement
                    report.initialize(nr_channels, subgrid_size, grid_size);
                    State states[2];
                    states[0] = powerSensor->read();

                    // Run subroutines
                    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                        unsigned int first_bl, last_bl, current_nr_baselines;
                        plan.initialize_job(nr_baselines, jobsize, bl, &first_bl, &last_bl, &current_nr_baselines);
                        if (current_nr_baselines == 0) continue;

                        // Initialize iteration
                        auto current_nr_subgrids  = plan.get_nr_subgrids(first_bl, current_nr_baselines);
                        const float *shift_ptr = shift.data();
                        void *wavenumbers_ptr  = wavenumbers.data();
                        void *spheroidal_ptr   = spheroidal.data();
                        void *aterm_ptr        = aterms.data();
                        void *metadata_ptr     = (void *) plan.get_metadata_ptr(first_bl);
                        void *uvw_ptr          = uvw.data(first_bl, 0);
                        void *visibilities_ptr = visibilities.data(first_bl, 0, 0);
                        void *subgrids_ptr     = subgrids.data(0, 0, 0, 0);
                        void *grid_ptr         = grid.data();

                        // Splitter kernel
                        if (w_step == 0.0) {
                           kernels.run_splitter(current_nr_subgrids, grid_size, subgrid_size, metadata_ptr, subgrids_ptr, grid_ptr);
                        } else {
                           kernels.run_splitter_wstack(current_nr_subgrids, grid_size, subgrid_size, metadata_ptr, subgrids_ptr, grid_ptr);
                        }

                        // FFT kernel
                        kernels.run_subgrid_fft(grid_size, subgrid_size, current_nr_subgrids, subgrids_ptr, FFTW_FORWARD);

                        // Degridder kernel
                        kernels.run_degridder(
                            current_nr_subgrids,
                            grid_size,
                            subgrid_size,
                            image_size,
                            w_step,
                            shift_ptr,
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
                        auto current_nr_timesteps = plan.get_nr_timesteps(first_bl, current_nr_baselines);
                        report.print(current_nr_timesteps, current_nr_subgrids);
                        #endif
                    } // end for bl

                    states[1] = powerSensor->read();
                    report.update_host(states[0], states[1]);

                    // Report performance
                    #if defined(REPORT_TOTAL)
                    auto total_nr_subgrids  = plan.get_nr_subgrids();
                    auto total_nr_timesteps = plan.get_nr_timesteps();
                    report.print_total(total_nr_timesteps, total_nr_subgrids);
                    auto total_nr_visibilities = plan.get_nr_visibilities();
                    report.print_visibilities(auxiliary::name_degridding, total_nr_visibilities);
                    #endif

                } catch (const std::invalid_argument& e) {
                    std::cerr << __func__ << ": invalid argument: "
                         << e.what() << std::endl;
                    exit(1);
                } catch (const std::exception& e) {
                    std::cerr << __func__ << ": caught exception: "
                         << e.what() << std::endl;
                    exit(2);
                } catch (...) {
                    std::cerr << __func__ << ": caught unknown exception" << std::endl;
                    exit(3);
                }
            } // end degridding

            void CPU::do_transform(
                DomainAtoDomainB direction,
                Array3D<std::complex<float>>& grid)
            {
                #if defined(DEBUG)
                std::cout << __func__ << std::endl;
                std::cout << "FFT (direction: " << direction << ")" << std::endl;
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
                    std::clog << std::endl;
                    #endif

                } catch (const std::exception& e) {
                    std::cerr << __func__ << " caught exception: "
                         << e.what() << std::endl;
                } catch (...) {
                    std::cerr << __func__ << " caught unknown exception" << std::endl;
                }
            } // end transform

        } // namespace cpu
    } // namespace proxy
} // namespace idg
