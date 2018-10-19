#include "KernelsInstance.h"

namespace idg {
    namespace kernel {

        void KernelsInstance::shift(
            Array3D<std::complex<float>>& data)
        {
            int nr_polarizations = data.get_z_dim();
            int height = data.get_y_dim();
            int width = data.get_x_dim();
            ASSERT(height == width);

            powersensor::State states[2];
            states[0] = powerSensor->read();

            std::complex<float> tmp13, tmp24;

            // Dimensions
            int n = height;
            int n2 = n / 2;

            // Interchange entries in 4 quadrants, 1 <--> 3 and 2 <--> 4
            for (int pol = 0; pol < nr_polarizations; pol++) {
                #pragma omp parallel for private(tmp13, tmp24) collapse(2)
                for (int i = 0; i < n2; i++) {
                    for (int k = 0; k < n2; k++) {
                        tmp13                 = data(pol, i, k);
                        data(pol, i, k)       = data(pol, i+n2, k+n2);
                        data(pol, i+n2, k+n2) = tmp13;

                        tmp24                 = data(pol, i+n2, k);
                        data(pol, i+n2, k)    = data(pol, i, k+n2);
                        data(pol, i, k+n2)    = tmp24;
                     }
                }
            }

            states[1] = powerSensor->read();
            report->update_fft_shift(states[0], states[1]);
        }

        void KernelsInstance::scale(
            Array3D<std::complex<float>>& data,
            std::complex<float> scale) const
        {
            int nr_polarizations = data.get_z_dim();
            int height = data.get_y_dim();
            int width = data.get_x_dim();

            powersensor::State states[2];
            states[0] = powerSensor->read();

            #pragma omp parallel for collapse(3)
            for (int pol = 0; pol < nr_polarizations; pol++) {
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        std::complex<float> value = data(pol, y, x);
                        data(pol, y, x) = std::complex<float>(
                            value.real() * scale.real(),
                            value.imag() * scale.imag());
                    }
                }
            }

            states[1] = powerSensor->read();
            report->update_fft_scale(states[0], states[1]);
        }

        void KernelsInstance::tile_backward(
            const int tile_size,
            const Grid& grid_src,
                  Grid& grid_dst) const
        {
            ASSERT(grid_src.bytes() == grid_dst.bytes());
            const int nr_correlations = grid_src.get_z_dim();
            const int height = grid_src.get_y_dim();
            const int width  = grid_dst.get_x_dim();
            ASSERT(height == width);
            const int grid_size = height;

            std::complex<float>* src_ptr = (std::complex<float> *) grid_src.data();
            std::complex<float>* dst_ptr = (std::complex<float> *) grid_dst.data();

            #pragma omp parallel for
            for (int pol = 0; pol < nr_correlations; pol++) {
                for (int y = 0; y < grid_size; y++) {
                    for (int x = 0; x < grid_size; x++) {
                        long src_idx = index_grid_tiling(tile_size, nr_correlations, grid_size, pol, y, x);
                        long dst_idx = index_grid(nr_correlations, grid_size, 0, pol, y, x);

                        dst_ptr[dst_idx] = src_ptr[src_idx];
                    }
                }
            }
        }

        void KernelsInstance::tile_forward(
            const int tile_size,
            const Grid& grid_src,
                  Grid& grid_dst) const
        {
            ASSERT(grid_src.bytes() == grid_dst.bytes());
            const int nr_correlations = grid_src.get_z_dim();
            const int height = grid_src.get_y_dim();
            const int width  = grid_dst.get_x_dim();
            ASSERT(height == width);
            const int grid_size = height;

            std::complex<float>* src_ptr = (std::complex<float> *) grid_src.data();
            std::complex<float>* dst_ptr = (std::complex<float> *) grid_dst.data();

            #pragma omp parallel for
            for (int pol = 0; pol < nr_correlations; pol++) {
                for (int y = 0; y < grid_size; y++) {
                    for (int x = 0; x < grid_size; x++) {
                        long src_idx = index_grid(nr_correlations, grid_size, 0, pol, y, x);
                        long dst_idx = index_grid_tiling(tile_size, nr_correlations, grid_size, pol, y, x);
                        dst_ptr[dst_idx] = src_ptr[src_idx];
                    }
                }
            }
        }

    } // namespace kernel
} // namespace idg
