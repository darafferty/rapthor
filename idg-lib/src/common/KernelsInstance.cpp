#include "KernelsInstance.h"

#include "Index.h"

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
            for (int pixel = 0; pixel < grid_size*grid_size; pixel++) {
                int y = pixel / grid_size;
                int x = pixel % grid_size;
                for (int pol = 0; pol < nr_correlations; pol++) {
                    long src_idx = index_grid_tiling(tile_size, grid_size, pol, y, x);
                    long dst_idx = index_grid(grid_size, 0, pol, y, x);

                    dst_ptr[dst_idx] = src_ptr[src_idx];
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
            for (int pixel = 0; pixel < grid_size*grid_size; pixel++) {
                int y = pixel / grid_size;
                int x = pixel % grid_size;
                for (int pol = 0; pol < nr_correlations; pol++) {
                    long src_idx = index_grid(grid_size, 0, pol, y, x);
                    long dst_idx = index_grid_tiling(tile_size, grid_size, pol, y, x);
                    dst_ptr[dst_idx] = src_ptr[src_idx];
                }
            }
        }

        void KernelsInstance::transpose_aterm(
            const Array4D<Matrix2x2<std::complex<float>>>& aterms_src,
                  Array4D<std::complex<float>>& aterms_dst) const
        {
            ASSERT(aterms_src.bytes() == aterms_dst.bytes());
            ASSERT(aterms_src.get_y_dim() == aterms_src.get_x_dim());
            ASSERT(aterms_dst.get_z_dim() == NR_CORRELATIONS);
            const unsigned int nr_stations  = aterms_src.get_w_dim();
            const unsigned int nr_timeslots = aterms_src.get_z_dim();
            const unsigned int subgrid_size = aterms_src.get_y_dim();

            #pragma omp parallel for
            for (unsigned int pixel = 0; pixel < subgrid_size*subgrid_size; pixel++) {
                for (unsigned int station = 0; station < nr_stations; station++) {
                    for (unsigned int timeslot = 0; timeslot < nr_timeslots; timeslot++) {
                        unsigned int y = pixel / subgrid_size;
                        unsigned int x = pixel % subgrid_size;
                        unsigned int term_nr = station * nr_timeslots + timeslot;

                        Matrix2x2<std::complex<float>> term = aterms_src(station, timeslot, y, x);
                        aterms_dst(term_nr, 0, y, x) = term.xx;
                        aterms_dst(term_nr, 1, y, x) = term.xy;
                        aterms_dst(term_nr, 2, y, x) = term.yx;
                        aterms_dst(term_nr, 3, y, x) = term.yy;
                    }
                }
            }
        }

        void KernelsInstance::print_memory_info() {
            auto memory_total = auxiliary::get_total_memory() / (float) 1024; // GBytes
            auto memory_used  = auxiliary::get_used_memory() / (float) 1024; // GBytes
            auto memory_free  = memory_total - memory_used;
            std::clog << "Host memory -> " << std::fixed << std::setprecision(1);
            std::clog << "total: " << memory_total << " Gb, ";
            std::clog << "used: "  << memory_used  << " Gb, ";
            std::clog << "free: "  << memory_free  << " Gb" << std::endl;
        }

    } // namespace kernel
} // namespace idg
