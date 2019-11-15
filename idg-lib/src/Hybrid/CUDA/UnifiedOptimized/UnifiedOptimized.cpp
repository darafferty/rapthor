#include <algorithm> // max_element

#include "UnifiedOptimized.h"


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
        namespace hybrid {

            // Constructor
            UnifiedOptimized::UnifiedOptimized(
                ProxyInfo info) :
                CUDA(info)
            {
                #if defined(DEBUG)
                std::cout << "UnifiedOptimized::" << __func__ << std::endl;
                #endif

                cpuProxy = new idg::proxy::cpu::Optimized();
                gpuProxy = new idg::proxy::cuda::Generic();
                gpuProxy->enable_unified_memory();

                // Increase the fraction of reserved memory
                set_fraction_reserved(0.4);
            }

            // Destructor
            UnifiedOptimized::~UnifiedOptimized() {
                #if defined(DEBUG)
                std::cout << "UnifiedOptimized::" << __func__ << std::endl;
                #endif

                delete cpuProxy;
                delete gpuProxy;
            }


            void UnifiedOptimized::do_transform(
                DomainAtoDomainB direction,
                Array3D<std::complex<float>>& grid)
            {
                #if defined(DEBUG)
                std::cout << "UnifiedOptimized::" << __func__ << std::endl;
                cout << "Transform direction: " << direction << endl;
                #endif

                cpuProxy->transform(direction, grid);
            } // end transform


            void UnifiedOptimized::do_gridding(
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
                gpuProxy->run_gridding(
                    plan, w_step, shift, cell_size, kernel_size, subgrid_size,
                    frequencies, visibilities, uvw, baselines,
                    grid_tiled, aterms, aterms_offsets, spheroidal);

                // Undo tiling
                #if ENABLE_TILING
                device.tile_backward(grid_size, tile_size, grid_tiled, grid);
                #endif
            } // end gridding


            void UnifiedOptimized::do_degridding(
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
                gpuProxy->run_degridding(
                    plan, w_step, shift, cell_size, kernel_size, subgrid_size,
                    frequencies, visibilities, uvw, baselines,
                    grid_tiled, aterms, aterms_offsets, spheroidal);

                // Undo tiling
                #if ENABLE_TILING
                device.tile_backward(grid_size, tile_size, grid_tiled, (Grid&) grid);
                #endif
            } // end degridding

        } // namespace hybrid
    } // namespace proxy
} // namespace idg

#include "UnifiedOptimizedC.h"
