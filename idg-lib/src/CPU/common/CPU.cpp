#include <vector>
#include <memory>
#include <climits>

#include "fftw3.h"

#include "CPU.h"

//#define DEBUG_COMPUTE_JOBSIZE

using namespace idg::kernel;
using namespace powersensor;

namespace idg {
    namespace proxy {
        namespace cpu {

            // Constructor
            CPU::CPU(
                std::vector<std::string> libraries):
                kernels(libraries),
                itsWTiles(0),
                itsWTilesBuffer(0)
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

                // Deallocate FFTWs internally allocated memory
                fftwf_cleanup();
            }

            std::shared_ptr<auxiliary::Memory> CPU::allocate_memory(size_t bytes)
            {
                return std::shared_ptr<auxiliary::Memory>(new auxiliary::AlignedMemory(bytes));
            }

            Plan* CPU::make_plan(
                const int kernel_size,
                const int subgrid_size,
                const int grid_size,
                const float cell_size,
                const Array1D<float>& frequencies,
                const Array2D<UVW<float>>& uvw,
                const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                const Array1D<unsigned int>& aterms_offsets,
                Plan::Options options)
            {
                if (supports_wtiles() && options.w_step != 0.0) {

                    //TODO call somewhere else
                    init_wtiles(subgrid_size);
                    options.nr_w_layers = INT_MAX;

                    return new Plan(
                        kernel_size,
                        subgrid_size,
                        grid_size,
                        cell_size,
                        frequencies,
                        uvw,
                        baselines,
                        aterms_offsets,
                        itsWTiles,
                        options
                    );
                } else {
                    return Proxy::make_plan(
                        kernel_size,
                        subgrid_size,
                        grid_size,
                        cell_size,
                        frequencies,
                        uvw,
                        baselines,
                        aterms_offsets,
                        options
                    );
                }
            }


            unsigned int CPU::compute_jobsize(
                const Plan& plan,
                const unsigned int nr_timesteps,
                const unsigned int nr_channels,
                const unsigned int subgrid_size)
            {
                auto nr_baselines = plan.get_nr_baselines();
                auto jobsize = nr_baselines;
                auto sizeof_visibilities = auxiliary::sizeof_visibilities(nr_baselines, nr_timesteps, nr_channels);

                // Make sure that every job will fit in memory
                do {
                    // Determine the maximum number of subgrids for this jobsize
                    auto max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);

                    // Determine the size of the subgrids for this jobsize
                    auto sizeof_subgrids = auxiliary::sizeof_subgrids(max_nr_subgrids, subgrid_size);

                    #if defined(DEBUG_COMPUTE_JOBSIZE)
                    std::clog << "size of subgrids: " << sizeof_subgrids << std::endl;
                    #endif

                    // Determine the amount of free memory
                    auto free_memory = auxiliary::get_free_memory(); // Mb
                    free_memory *= 1024 * 1024; // Byte

                    // Limit the amount of memory used for subgrids
                    free_memory *= m_fraction_memory_subgrids;

                    // Determine whether to proceed with the current jobsize
                    if (sizeof_subgrids < sizeof_visibilities &&
                        sizeof_subgrids < free_memory &&
                        sizeof_subgrids < m_max_bytes_subgrids) {
                        break;
                    }

                    // Reduce jobsize
                    jobsize *= 0.8;
                } while (jobsize > 1);

                #if defined(DEBUG_COMPUTE_JOBSIZE)
                std::clog << "jobsize: " << jobsize << std::endl;
                #endif

                return jobsize;
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
                const Array2D<UVW<float>>& uvw,
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
                auto nr_timesteps = visibilities.get_y_dim();
                auto nr_channels  = visibilities.get_x_dim();
                auto grid_size    = grid.get_x_dim();
                auto image_size   = cell_size * grid_size;
                auto nr_stations  = aterms.get_z_dim();

                try {
                    auto jobsize = compute_jobsize(plan, nr_timesteps, nr_channels, subgrid_size);

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
                        void *aterm_idx_ptr    = (void *) plan.get_aterm_indices_ptr();
                        void *avg_aterm_ptr    = m_avg_aterm_correction.size() ? m_avg_aterm_correction.data() : nullptr;
                        void *metadata_ptr     = (void *) plan.get_metadata_ptr(first_bl);
                        void *uvw_ptr          = uvw.data(0, 0);
                        void *visibilities_ptr = visibilities.data(0, 0, 0);
                        void *subgrids_ptr     = subgrids.data(0, 0, 0, 0);
                        void *grid_ptr         = grid.data();

                        // Gridder kernel
                        kernels.run_gridder(
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
                            aterm_idx_ptr,
                            avg_aterm_ptr,
                            metadata_ptr,
                            subgrids_ptr);

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
                        auto current_nr_timesteps = plan.get_nr_timesteps(first_bl, current_nr_baselines);
                        report.print(current_nr_timesteps, current_nr_subgrids);
                    } // end for bl

                    states[1] = powerSensor->read();
                    report.update_host(states[0], states[1]);

                    // Performance report
                    auto total_nr_subgrids  = plan.get_nr_subgrids();
                    auto total_nr_timesteps = plan.get_nr_timesteps();
                    report.print_total(total_nr_timesteps, total_nr_subgrids);
                    auto total_nr_visibilities = plan.get_nr_visibilities();
                    report.print_visibilities(auxiliary::name_gridding, total_nr_visibilities);

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
                const Array2D<UVW<float>>& uvw,
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
                auto nr_timesteps = visibilities.get_y_dim();
                auto nr_channels  = visibilities.get_x_dim();
                auto grid_size    = grid.get_x_dim();
                auto image_size   = cell_size * grid_size;
                auto nr_stations  = aterms.get_z_dim();

                try {
                    auto jobsize = compute_jobsize(plan, nr_timesteps, nr_channels, subgrid_size);

                    // Allocate memory for subgrids
                    int max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);
                    Array4D<std::complex<float>> subgrids(max_nr_subgrids, nr_polarizations, subgrid_size, subgrid_size);

                    WTileUpdateSet wtile_initialize_set = plan.get_wtile_initialize_set();

                    // initialize wtiles
                    // the front entry of the wtile_initialize_set will be initialized, but it will remain in the queue
                    //
                    if (plan.get_use_wtiles()) {
                        WTileUpdateInfo &wtile_initialize_info = wtile_initialize_set.front();
                        kernels.run_splitter_wtiles_from_grid(
                            grid_size,
                            subgrid_size,
                            image_size,
                            w_step,
                            wtile_initialize_info.wtile_ids.size(),
                            wtile_initialize_info.wtile_ids.data(),
                            wtile_initialize_info.wtile_coordinates.data(),
                            itsWTilesBuffer.data(),
                            grid.data());
                    }

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
                        void *aterm_idx_ptr    = (void *) plan.get_aterm_indices_ptr();
                        void *metadata_ptr     = (void *) plan.get_metadata_ptr(first_bl);
                        void *uvw_ptr          = uvw.data(0, 0);
                        void *visibilities_ptr = visibilities.data(0, 0, 0);
                        void *subgrids_ptr     = subgrids.data(0, 0, 0, 0);
                        void *grid_ptr         = grid.data();
                        void *wtiles_ptr       = itsWTilesBuffer.data();

                        // Splitter kernel
                        if (w_step == 0.0) {
                           kernels.run_splitter(current_nr_subgrids, grid_size, subgrid_size, metadata_ptr, subgrids_ptr, grid_ptr);
                        } else if (plan.get_use_wtiles()) {
                            auto subgrid_offset = plan.get_subgrid_offset(bl);
                            kernels.run_splitter_wtiles(
                                current_nr_subgrids, grid_size, subgrid_size, image_size, w_step,
                                subgrid_offset, wtile_initialize_set, wtiles_ptr, metadata_ptr, subgrids_ptr, grid_ptr);
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
                            aterm_idx_ptr,
                            metadata_ptr,
                            subgrids_ptr);

                        // Performance reporting
                        auto current_nr_timesteps = plan.get_nr_timesteps(first_bl, current_nr_baselines);
                        report.print(current_nr_timesteps, current_nr_subgrids);
                    } // end for bl

                    states[1] = powerSensor->read();
                    report.update_host(states[0], states[1]);

                    // Report performance
                    auto total_nr_subgrids  = plan.get_nr_subgrids();
                    auto total_nr_timesteps = plan.get_nr_timesteps();
                    report.print_total(total_nr_timesteps, total_nr_subgrids);
                    auto total_nr_visibilities = plan.get_nr_visibilities();
                    report.print_visibilities(auxiliary::name_degridding, total_nr_visibilities);

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


            void CPU::do_calibrate_init(
                std::vector<std::unique_ptr<Plan>> &&plans,
                float w_step,
                Array1D<float> &&shift,
                float cell_size,
                unsigned int kernel_size,
                unsigned int subgrid_size,
                const Array1D<float> &frequencies,
                Array4D<Visibility<std::complex<float>>> &&visibilities,
                Array4D<Visibility<float>> &&weights,
                Array3D<UVW<float>> &&uvw,
                Array2D<std::pair<unsigned int,unsigned int>> &&baselines,
                const Grid& grid,
                const Array2D<float>& spheroidal)
            {
                Array1D<float> wavenumbers = compute_wavenumbers(frequencies);

                // Arguments
                auto nr_antennas  = plans.size();
                auto grid_size    = grid.get_x_dim();
                auto image_size   = cell_size * grid_size;
                auto nr_baselines = visibilities.get_z_dim();
                auto nr_timesteps = visibilities.get_y_dim();
                auto nr_channels  = visibilities.get_x_dim();

                // Allocate subgrids for all antennas
                std::vector<Array4D<std::complex<float>>> subgrids;
                subgrids.reserve(nr_antennas);

                // Allocate phasors for all antennas
                std::vector<Array4D<std::complex<float>>> phasors;
                phasors.reserve(nr_antennas);

                std::vector<int> max_nr_timesteps;
                max_nr_timesteps.reserve(nr_antennas);

                // Start performance measurement
                report.initialize();
                powersensor::State states[2];
                states[0] = powerSensor->read();

                // Create subgrids for every antenna
                for (unsigned int antenna_nr = 0; antenna_nr < nr_antennas; antenna_nr++)
                {
                    // Allocate subgrids for current antenna
                    int nr_subgrids = plans[antenna_nr]->get_nr_subgrids();
                    Array4D<std::complex<float>> subgrids_(nr_subgrids, nr_polarizations, subgrid_size, subgrid_size);

                    WTileUpdateSet wtile_initialize_set = plans[antenna_nr]->get_wtile_initialize_set();

                    // initialize wtiles
                    // the front entry of the wtile_initialize_set will be initialized, but it will remain in the queue
                    //
                    if (plans[antenna_nr]->get_use_wtiles()) {
                        WTileUpdateInfo &wtile_initialize_info = wtile_initialize_set.front();
                        kernels.run_splitter_wtiles_from_grid(
                            grid_size,
                            subgrid_size,
                            image_size,
                            w_step,
                            wtile_initialize_info.wtile_ids.size(),
                            wtile_initialize_info.wtile_ids.data(),
                            wtile_initialize_info.wtile_coordinates.data(),
                            itsWTilesBuffer.data(),
                            grid.data());
                    }


                    // Get data pointers
                    const float *shift_ptr = shift.data();
                    void *metadata_ptr     = (void *) plans[antenna_nr]->get_metadata_ptr();
                    void *subgrids_ptr     = subgrids_.data();
                    void *grid_ptr         = grid.data();

                    // Splitter kernel
                    if (w_step == 0.0) {
                        kernels.run_splitter(nr_subgrids, grid_size, subgrid_size, metadata_ptr, subgrids_ptr, grid_ptr);
                    } else if (plans[antenna_nr]->get_use_wtiles()) {
                        for(int subgrid_index = 0; subgrid_index < nr_subgrids; )
                        {
                            if (wtile_initialize_set.front().subgrid_index == subgrid_index)
                            {
                                wtile_initialize_set.pop_front();
                                WTileUpdateInfo &wtile_initialize_info = wtile_initialize_set.front();
                                kernels.run_splitter_wtiles_from_grid(
                                    grid_size,
                                    subgrid_size,
                                    image_size,
                                    w_step,
                                    wtile_initialize_info.wtile_ids.size(),
                                    wtile_initialize_info.wtile_ids.data(),
                                    wtile_initialize_info.wtile_coordinates.data(),
                                    itsWTilesBuffer.data(),
                                    grid_ptr);
                            }

                            int nr_subgrids_ = nr_subgrids - subgrid_index;
                            if (wtile_initialize_set.front().subgrid_index - subgrid_index < nr_subgrids_)
                            {
                                nr_subgrids_ = wtile_initialize_set.front().subgrid_index - subgrid_index;
                            }

                            kernels.run_splitter_subgrids_from_wtiles(
                                nr_subgrids_,
                                grid_size,
                                subgrid_size,
                                &static_cast<Metadata*>(metadata_ptr)[subgrid_index],
                                &static_cast<std::complex<float>*>(subgrids_ptr)[subgrid_index * subgrid_size * subgrid_size * NR_CORRELATIONS],
                                itsWTilesBuffer.data());

                            subgrid_index += nr_subgrids_;
                        }
                    } else {
                        kernels.run_splitter_wstack(nr_subgrids, grid_size, subgrid_size, metadata_ptr, subgrids_ptr, grid_ptr);
                    }

                    // FFT kernel
                    kernels.run_subgrid_fft(grid_size, subgrid_size, nr_subgrids, subgrids_ptr, FFTW_FORWARD);

                    // Apply spheroidal
                    for (int i = 0; i < nr_subgrids; i++) {
                        for (unsigned int pol = 0; pol < nr_polarizations; pol++) {
                            for (unsigned int j = 0; j < subgrid_size; j++) {
                                for (unsigned int k = 0; k < subgrid_size; k++) {
                                    unsigned int y = (j + (subgrid_size/2)) % subgrid_size;
                                    unsigned int x = (k + (subgrid_size/2)) % subgrid_size;
                                    subgrids_(i, pol, y, x) *= spheroidal(j,k);
                                }
                            }
                        }
                    }

                    // Store subgrids for current antenna
                    subgrids.push_back(std::move(subgrids_));

                    // Get max number of timesteps for any subgrid
                    auto max_nr_timesteps_ = plans[antenna_nr]->get_max_nr_timesteps_subgrid();
                    max_nr_timesteps.push_back(max_nr_timesteps_);

                    // Allocate phasors for current antenna
                    Array4D<std::complex<float>> phasors_(nr_subgrids * max_nr_timesteps_, nr_channels, subgrid_size, subgrid_size);

                    // Get data pointers
                    void *wavenumbers_ptr  = wavenumbers.data();
                    void *uvw_ptr          = uvw.data(antenna_nr);
                    void *phasors_ptr      = phasors_.data();

                    // Compute phasors
                    kernels.run_phasor(
                        nr_subgrids,
                        grid_size,
                        subgrid_size,
                        image_size,
                        w_step,
                        shift_ptr,
                        max_nr_timesteps_,
                        nr_channels,
                        uvw_ptr,
                        wavenumbers_ptr,
                        metadata_ptr,
                        phasors_ptr);

                    // Store phasors for current antenna
                    phasors.push_back(std::move(phasors_));
                } // end for antennas

                // End performance measurement
                states[1] = powerSensor->read();
                report.update_host(states[0], states[1]);
                report.print_total(0, 0);

                // Set calibration state member variables
                m_calibrate_state = {
                    std::move(plans),
                    w_step,
                    std::move(shift),
                    cell_size,
                    image_size,
                    kernel_size,
                    (unsigned int) grid_size,
                    subgrid_size,
                    (unsigned int) nr_baselines,
                    (unsigned int) nr_timesteps,
                    (unsigned int) nr_channels,
                    std::move(wavenumbers),
                    std::move(visibilities),
                    std::move(weights),
                    std::move(uvw),
                    std::move(baselines),
                    std::move(subgrids),
                    std::move(phasors),
                    std::move(max_nr_timesteps)
                };
            }

            void CPU::do_calibrate_update(
                const int antenna_nr,
                const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                const Array4D<Matrix2x2<std::complex<float>>>& aterm_derivatives,
                Array3D<double>& hessian,
                Array2D<double>& gradient,
                double &residual)
            {
                // Arguments
                auto nr_subgrids   = m_calibrate_state.plans[antenna_nr]->get_nr_subgrids();
                auto nr_channels   = m_calibrate_state.wavenumbers.get_x_dim();
                auto nr_terms      = aterm_derivatives.get_z_dim();
                auto subgrid_size  = aterms.get_y_dim();
                auto nr_stations   = aterms.get_z_dim();
                auto nr_timeslots  = aterms.get_w_dim();

                // Performance measurement
                if (antenna_nr == 0) {
                    report.initialize(nr_channels, subgrid_size, 0, nr_terms);
                }

                // Data pointers
                auto shift_ptr                     = m_calibrate_state.shift.data();
                auto wavenumbers_ptr               = m_calibrate_state.wavenumbers.data();
                idg::float2 *aterm_ptr             = (idg::float2*) aterms.data();
                idg::float2 * aterm_derivative_ptr = (idg::float2*) aterm_derivatives.data();
                auto aterm_idx_ptr                 = m_calibrate_state.plans[antenna_nr]->get_aterm_indices_ptr();
                auto metadata_ptr                  = m_calibrate_state.plans[antenna_nr]->get_metadata_ptr();
                auto uvw_ptr                       = m_calibrate_state.uvw.data(antenna_nr);
                idg::float2 *visibilities_ptr      = (idg::float2*) m_calibrate_state.visibilities.data(antenna_nr);
                float *weights_ptr                 = (float*) m_calibrate_state.weights.data(antenna_nr);
                idg::float2 *subgrids_ptr          = (idg::float2*) m_calibrate_state.subgrids[antenna_nr].data();
                idg::float2 *phasors_ptr           = (idg::float2*) m_calibrate_state.phasors[antenna_nr].data();
                double *hessian_ptr                = hessian.data();
                double *gradient_ptr               = gradient.data();
                double *residual_ptr               = &residual;

                int max_nr_timesteps       = m_calibrate_state.max_nr_timesteps[antenna_nr];

                // Run calibration update step
                kernels.run_calibrate(
                    nr_subgrids,
                    m_calibrate_state.grid_size,
                    m_calibrate_state.subgrid_size,
                    m_calibrate_state.image_size,
                    m_calibrate_state.w_step,
                    shift_ptr,
                    max_nr_timesteps,
                    nr_channels,
                    nr_terms,
                    nr_stations,
                    nr_timeslots,
                    uvw_ptr,
                    wavenumbers_ptr,
                    visibilities_ptr,
                    weights_ptr,
                    aterm_ptr,
                    aterm_derivative_ptr,
                    aterm_idx_ptr,
                    metadata_ptr,
                    subgrids_ptr,
                    phasors_ptr,
                    hessian_ptr,
                    gradient_ptr,
                    residual_ptr);

                // Performance reporting
                auto current_nr_subgrids  = nr_subgrids;
                auto current_nr_timesteps = m_calibrate_state.plans[antenna_nr]->get_nr_timesteps();
                auto current_nr_visibilities = current_nr_timesteps * nr_channels;
                report.update_total(current_nr_subgrids, current_nr_timesteps, current_nr_visibilities);
            }

            void CPU::do_calibrate_finish()
            {
                // Performance reporting
                auto nr_antennas  = m_calibrate_state.plans.size();
                auto total_nr_timesteps = 0;
                auto total_nr_subgrids  = 0;
                for (unsigned int antenna_nr = 0; antenna_nr < nr_antennas; antenna_nr++) {
                    total_nr_timesteps += m_calibrate_state.plans[antenna_nr]->get_nr_timesteps();
                    total_nr_subgrids  += m_calibrate_state.plans[antenna_nr]->get_nr_subgrids();
                }
                report.print_total(total_nr_timesteps, total_nr_subgrids);
                report.print_visibilities(auxiliary::name_calibrate);
            }

            void CPU::do_calibrate_init_hessian_vector_product()
            {
                m_calibrate_state.hessian_vector_product_visibilities = Array3D<Visibility<std::complex<float>>>(
                    m_calibrate_state.nr_baselines,
                    m_calibrate_state.nr_timesteps,
                    m_calibrate_state.nr_channels
                );
                m_calibrate_state.hessian_vector_product_visibilities.zero();
            }

            void CPU::do_calibrate_update_hessian_vector_product1(
                const int antenna_nr,
                const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                const Array4D<Matrix2x2<std::complex<float>>>& aterm_derivatives,
                const Array2D<float>& parameter_vector)
            {
                // TODO
#if 0
                // Arguments
                auto nr_subgrids   = m_calibrate_state.plans[antenna_nr]->get_nr_subgrids();
                auto nr_channels   = m_calibrate_state.wavenumbers.get_x_dim();
                auto nr_terms      = aterm_derivatives.get_z_dim();
                auto subgrid_size  = aterms.get_y_dim();
                auto nr_stations   = aterms.get_z_dim();
                auto nr_timeslots  = aterms.get_w_dim();

                // Performance measurement
                if (antenna_nr == 0) {
                    report.initialize(nr_channels, subgrid_size, 0, nr_terms);
                }

                // Data pointers
                auto shift_ptr                     = m_calibrate_state.shift.data();
                auto wavenumbers_ptr               = m_calibrate_state.wavenumbers.data();
                idg::float2 *aterm_ptr             = (idg::float2*) aterms.data();
                idg::float2 * aterm_derivative_ptr = (idg::float2*) aterm_derivatives.data();
                auto aterm_idx_ptr                 = m_calibrate_state.plans[antenna_nr]->get_aterm_indices_ptr();
                auto metadata_ptr                  = m_calibrate_state.plans[antenna_nr]->get_metadata_ptr();
                auto uvw_ptr                       = m_calibrate_state.uvw.data(antenna_nr);
                idg::float2 *visibilities_ptr      = (idg::float2*) m_calibrate_state.visibilities.data(antenna_nr);
                float *weights_ptr                 = (float*) m_calibrate_state.weights.data(antenna_nr);
                idg::float2 *subgrids_ptr          = (idg::float2*) m_calibrate_state.subgrids[antenna_nr].data();
                idg::float2 *phasors_ptr           = (idg::float2*) m_calibrate_state.phasors[antenna_nr].data();
                float *parameter_vector_ptr        = parameter_vector.data();

                int max_nr_timesteps       = m_calibrate_state.max_nr_timesteps[antenna_nr];


                kernels.run_calibrate_hessian_vector_product1(antenna_nr, aterms, aterm_derivatives, parameter_vector);
#endif
            }

            void CPU::do_calibrate_update_hessian_vector_product2(
                const int station_nr,
                const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                const Array4D<Matrix2x2<std::complex<float>>>& derivative_aterms,
                Array2D<float>& parameter_vector)
            {
                kernels.run_calibrate_hessian_vector_product2(station_nr, aterms, derivative_aterms, parameter_vector);
            }

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
                    report.print_total();
                    std::clog << std::endl;

                } catch (const std::exception& e) {
                    std::cerr << __func__ << " caught exception: "
                         << e.what() << std::endl;
                } catch (...) {
                    std::cerr << __func__ << " caught unknown exception" << std::endl;
                }
            } // end transform

            void CPU::init_wtiles(int subgrid_size)
            {
                if (itsWTilesBuffer.size() == 0) {
                    itsWTiles = WTiles(NR_WTILES);
                    itsWTilesBuffer = std::vector<std::complex<float>>(NR_WTILES * (WTILE_SIZE+subgrid_size)*(WTILE_SIZE+subgrid_size)*NR_CORRELATIONS);
                }
            }



        } // namespace cpu
    } // namespace proxy
} // namespace idg
