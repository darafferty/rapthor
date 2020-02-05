#include <algorithm> // max_element

#include "Unified.h"
#include "InstanceCUDA.h"

using namespace std;
using namespace idg::kernel::cuda;
using namespace powersensor;


namespace idg {
    namespace proxy {
        namespace cuda {

            // Constructor
            Unified::Unified(
                ProxyInfo info) :
                Generic(info)
            {
                #if defined(DEBUG)
                cout << "Unified::" << __func__ << endl;
                #endif

                // Increase the fraction of reserved memory
                set_fraction_reserved(0.4);

                // Enable unified memory
                enable_unified_memory();
            }

            // Destructor
            Unified::~Unified() {
                #if defined(DEBUG)
                std::cout << "Unified::" << __func__ << std::endl;
                #endif
            }


            void Unified::do_transform(
                DomainAtoDomainB direction,
                Array3D<std::complex<float>>& grid)
            {
                #if defined(DEBUG)
                std::cout << "Unified::" << __func__ << std::endl;
                cout << "Transform direction: " << direction << endl;
                #endif

                // TODO: fix this method
                // (1) does the m_grid_tiled need to be untiled before use?
                // (2) even though m_grid_tiled is Unified Memory, cuFFT fails

                // Constants
                auto nr_correlations = grid.get_z_dim();;
                auto grid_size       = grid.get_x_dim();

                // Load device
                InstanceCUDA &device = get_device(0);

                // Get UnifiedMemory object for grid data
                cu::UnifiedMemory u_grid(m_grid_tiled->data(), m_grid_tiled->bytes());

                // Initialize
                cu::Stream& stream = device.get_execute_stream();
                device.set_context();

                // Performance measurements
                report.initialize(0, 0, grid_size);
                device.set_report(report);
                PowerRecord powerRecords[2];
                State powerStates[4];
                powerStates[0] = hostPowerSensor->read();
                powerStates[2] = device.measure();

                // Perform fft shift
                double time_shift = -omp_get_wtime();
                device.shift(grid);
                time_shift += omp_get_wtime();

                // Execute fft
                device.measure(powerRecords[0], stream);
                device.launch_grid_fft_unified(grid_size, nr_correlations, grid, direction);
                device.measure(powerRecords[1], stream);
                stream.synchronize();

                // Perform fft shift
                time_shift = -omp_get_wtime();
                device.shift(grid);
                time_shift += omp_get_wtime();

                // Perform fft scaling
                double time_scale = -omp_get_wtime();
                complex<float> scale = complex<float>(2.0/(grid_size*grid_size), 0);
                if (direction == FourierDomainToImageDomain) {
                    device.scale(grid, scale);
                }
                time_scale += omp_get_wtime();

                // End measurements
                stream.synchronize();
                powerStates[1] = hostPowerSensor->read();
                powerStates[3] = device.measure();

                // Report performance
                report.update_grid_fft(powerRecords[0].state, powerRecords[1].state);
                report.update_fft_shift(time_shift);
                report.update_fft_scale(time_scale);
                report.print_total();
                report.print_device(powerStates[2], powerStates[3]);
                clog << endl;
            } // end transform


            void Unified::do_gridding(
                const Plan& plan,
                const float w_step, // in lambda
                const Array1D<float>& shift,
                const float cell_size,
                const unsigned int kernel_size, // full width in pixels
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
                std::cout << "Unified::" << __func__ << std::endl;
                #endif
                if (!m_use_unified_memory) {
                    throw std::runtime_error("Unified memory needs to be enabled!");
                }

                #if defined(DEBUG)
                std::clog << "### Initialize gridding" << std::endl;
                #endif
                CUDA::initialize(
                    plan, w_step, shift, cell_size, kernel_size, subgrid_size,
                    frequencies, visibilities, uvw, baselines,
                    aterms, aterms_offsets, spheroidal);

                #if defined(DEBUG)
                std::clog << "### Run gridding" << std::endl;
                #endif
                auto grid_ptr = m_enable_tiling ? m_grid_tiled.get() : m_grid.get();
                Generic::run_gridding(
                    plan, w_step, shift, cell_size, kernel_size, subgrid_size,
                    frequencies, visibilities, uvw, baselines,
                    *grid_ptr, aterms, aterms_offsets, spheroidal);
            } // end gridding


            void Unified::do_degridding(
                const Plan& plan,
                const float w_step, // in lambda
                const Array1D<float>& shift,
                const float cell_size,
                const unsigned int kernel_size, // full width in pixels
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
                std::cout << "Unified::" << __func__ << std::endl;
                #endif
                if (!m_use_unified_memory) {
                    throw std::runtime_error("Unified memory needs to be enabled!");
                }

                #if defined(DEBUG)
                std::clog << "### Initialize degridding" << std::endl;
                #endif
                CUDA::initialize(
                    plan, w_step, shift, cell_size, kernel_size, subgrid_size,
                    frequencies, visibilities, uvw, baselines,
                    aterms, aterms_offsets, spheroidal);

                #if defined(DEBUG)
                std::clog << "### Run degridding" << std::endl;
                #endif
                auto grid_ptr = m_enable_tiling ? m_grid_tiled.get() : m_grid.get();
                Generic::run_degridding(
                    plan, w_step, shift, cell_size, kernel_size, subgrid_size,
                    frequencies, visibilities, uvw, baselines,
                    *grid_ptr, aterms, aterms_offsets, spheroidal);
            } // end degridding


            void Unified::set_grid(
                std::shared_ptr<Grid> grid)
            {
                m_grid = grid;

                if (m_enable_tiling) {
                    InstanceCUDA &device = get_device(0);
                    auto grid_size = m_grid->get_x_dim();
                    auto tile_size = device.get_tile_size_grid();
                    cu::UnifiedMemory* u_grid_tiled = new cu::UnifiedMemory(m_grid->bytes());
                    auto grid_tiled = new Grid(*u_grid_tiled, grid->shape());
                    m_grid_tiled.reset(grid_tiled);
                    device.tile_forward(grid_size, tile_size, *grid, *m_grid_tiled);
                }
            }


            std::shared_ptr<Grid> Unified::get_grid()
            {
                if (m_enable_tiling) {
                    InstanceCUDA &device = get_device(0);
                    auto grid_size = m_grid->get_x_dim();
                    auto tile_size = device.get_tile_size_grid();
                    device.tile_backward(grid_size, tile_size, *m_grid_tiled, *m_grid);
                }

                return m_grid;
            }

        } // namespace cuda
    } // namespace proxy
} // namespace idg

#include "UnifiedC.h"
