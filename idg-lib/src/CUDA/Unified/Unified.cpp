#include <algorithm> // max_element

#include "Unified.h"
#include "InstanceCUDA.h"

using namespace std;
using namespace idg::kernel::cuda;
using namespace powersensor;


/*
 * Option to enable/disable reordering of the grid
 * to the host grid format, rather than the tiled
 * format used in the adder and splitter kernels.
 */
#define ENABLE_TILING 1


namespace idg {
    namespace proxy {
        namespace cuda {

            // Constructor
            Unified::Unified(
                ProxyInfo info) :
                CUDA(info)
            {
                #if defined(DEBUG)
                cout << "Unified::" << __func__ << endl;
                #endif

                gpuProxy = new idg::proxy::cuda::Generic();
                gpuProxy->enable_unified_memory();

                // Initialize host PowerSensor
                hostPowerSensor = get_power_sensor(sensor_host);

                // Increase the fraction of reserved memory
                set_fraction_reserved(0.4);
            }

            // Destructor
            Unified::~Unified() {
                #if defined(DEBUG)
                std::cout << "Unified::" << __func__ << std::endl;
                #endif

                delete gpuProxy;
                delete hostPowerSensor;
            }


            void Unified::do_transform(
                DomainAtoDomainB direction,
                Array3D<std::complex<float>>& grid)
            {
                #if defined(DEBUG)
                std::cout << "Unified::" << __func__ << std::endl;
                cout << "Transform direction: " << direction << endl;
                #endif

                // Constants
                auto nr_correlations = grid.get_z_dim();;
                auto grid_size       = grid.get_x_dim();

                // Load device
                InstanceCUDA &device = get_device(0);

                // Free device memory
                device.free_device_memory();

                // Get UnifiedMemory object for grid data
                cu::UnifiedMemory u_grid(grid.data(), grid.bytes());

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
				#if ENABLE_MEM_ADVISE
                u_grid.set_advice(CU_MEM_ADVISE_SET_ACCESSED_BY);
				#endif
                device.shift(grid);
                time_shift += omp_get_wtime();

                // Execute fft
                device.measure(powerRecords[0], stream);
				#if ENABLE_MEM_ADVISE
                u_grid.set_advice(CU_MEM_ADVISE_SET_ACCESSED_BY, device.get_device());
				#endif
                device.launch_fft_unified(grid_size, nr_correlations, grid, direction);
				#if ENABLE_MEM_ADVISE
                u_grid.set_advice(CU_MEM_ADVISE_SET_ACCESSED_BY);
				#endif
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
                std::cout << "UnifiedOptimized::" << __func__ << std::endl;
                #endif

                InstanceCUDA& device = get_device(0);

                // Arguments
                auto grid_size = grid.get_x_dim();
                auto tile_size = device.get_tile_size_grid();

                // Apply tiling
                cu::UnifiedMemory u_grid(grid.bytes());
                Grid grid_tiled(u_grid, grid.shape());
                #if ENABLE_TILING
                device.tile_forward(grid_size, tile_size, grid, grid_tiled);
                #endif

                // Run gridding
                gpuProxy->gridding(
                    plan, w_step, shift, cell_size, kernel_size, subgrid_size,
                    frequencies, visibilities, uvw, baselines,
                    grid_tiled, aterms, aterms_offsets, spheroidal);

                // Undo tiling
                #if ENABLE_TILING
                device.tile_backward(grid_size, tile_size, grid_tiled, grid);
                #endif
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
                std::cout << "UnifiedOptimized::" << __func__ << std::endl;
                #endif

                InstanceCUDA& device = get_device(0);

                // Arguments
                auto grid_size = grid.get_x_dim();
                auto tile_size = get_device(0).get_tile_size_grid();

                // Apply tiling
                cu::UnifiedMemory u_grid(grid.bytes());
                Grid grid_tiled(u_grid, grid.shape());
                #if ENABLE_TILING
                device.tile_forward(grid_size, tile_size, grid, grid_tiled);
                #endif

                // Run degridding
                gpuProxy->degridding(
                    plan, w_step, shift, cell_size, kernel_size, subgrid_size,
                    frequencies, visibilities, uvw, baselines,
                    grid_tiled, aterms, aterms_offsets, spheroidal);

                // Undo tiling
                #if ENABLE_TILING
                device.tile_backward(grid_size, tile_size, grid_tiled, (Grid&) grid);
                #endif
            } // end degridding

        } // namespace cuda
    } // namespace proxy
} // namespace idg

#include "UnifiedC.h"
