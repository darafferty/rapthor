#include "Kernels.h"

namespace idg {
    namespace kernel {

        /*
            Flop and byte count
        */
        uint64_t Kernels::flops_gridder(
            uint64_t nr_channels,
            uint64_t nr_timesteps,
            uint64_t nr_subgrids) const
        {
            uint64_t subgrid_size = mConstants.get_subgrid_size();
            uint64_t nr_correlations = mConstants.get_nr_correlations();

            // Number of flops per visibility
            uint64_t flops_per_visibility = 0;
            flops_per_visibility += 5; // phase index
            flops_per_visibility += 5; // phase offset
            flops_per_visibility += nr_channels * 2; // phase
            #if defined(REPORT_OPS)
            flops_per_visibility += nr_channels * 2; // phasor
            #endif
            flops_per_visibility += nr_channels * nr_correlations * 8; // update

            // Number of flops per subgrid
            uint64_t flops_per_subgrid = 0;
            flops_per_subgrid += nr_correlations * 30; // aterm
            flops_per_subgrid += nr_correlations * 2; // spheroidal
            flops_per_subgrid += 6; // shift

            // Total number of flops
            uint64_t flops_total = 0;
            flops_total += nr_timesteps * subgrid_size * subgrid_size * flops_per_visibility;
            flops_total += nr_subgrids  * subgrid_size * subgrid_size * flops_per_subgrid;
            return flops_total;
        }

        uint64_t Kernels::bytes_gridder(
            uint64_t nr_channels,
            uint64_t nr_timesteps,
            uint64_t nr_subgrids) const
        {
            uint64_t subgrid_size = mConstants.get_subgrid_size();
            uint64_t nr_correlations = mConstants.get_nr_correlations();

            // Number of bytes per uvw coordinate
            uint64_t bytes_per_uvw = 0;
            bytes_per_uvw += 1ULL * 3 * sizeof(float); // read uvw

            // Number of bytes per visibility
            uint64_t bytes_per_vis = 0;
            bytes_per_vis += 1ULL * nr_channels * nr_correlations * 2 * sizeof(float); // read visibilities

            // Number of bytes per pixel
            uint64_t bytes_per_pix = 0;
            bytes_per_pix += 1ULL * nr_correlations * 2 * sizeof(float); // read pixel
            bytes_per_pix += 1ULL * nr_correlations * 2 * sizeof(float); // write pixel

            // Number of bytes per aterm
            uint64_t bytes_per_aterm = 0;
            bytes_per_aterm += 1ULL * 2 * nr_correlations * 2 * sizeof(float); // read aterm

            // Number of bytes per spheroidal
            uint64_t bytes_per_spheroidal = 0;
            bytes_per_spheroidal += 1ULL * sizeof(float); // read spheroidal

            // Total number of bytes
            uint64_t bytes_total = 0;
            bytes_total += 1ULL * nr_timesteps * bytes_per_uvw;
            bytes_total += 1ULL * nr_timesteps * bytes_per_vis;
            bytes_total += 1ULL * nr_subgrids * subgrid_size * subgrid_size * bytes_per_pix;
            bytes_total += 1ULL * nr_subgrids * subgrid_size * subgrid_size * bytes_per_aterm;
            bytes_total += 1ULL * nr_subgrids * subgrid_size * subgrid_size * bytes_per_spheroidal;
            return bytes_total;
        }

        uint64_t Kernels::flops_degridder(
            uint64_t nr_channels,
            uint64_t nr_timesteps,
            uint64_t nr_subgrids) const
        {
            return flops_gridder(nr_channels, nr_timesteps, nr_subgrids);
        }

        uint64_t Kernels::bytes_degridder(
            uint64_t nr_channels,
            uint64_t nr_timesteps,
            uint64_t nr_subgrids) const
        {
            return bytes_gridder(nr_channels, nr_timesteps, nr_subgrids);
        }

        uint64_t Kernels::flops_fft(
            uint64_t size,
            uint64_t batch) const
        {
            uint64_t nr_correlations = mConstants.get_nr_correlations();
            // Pseudo number of flops:
            // return 1ULL * 5 * batch * nr_correlations * size * size * log2(size * size);
            // Estimated number of flops based on fftwf_flops, which seems to
            // return the number of simd instructions, not scalar flops.
            return 1ULL * 4 * batch * nr_correlations * size * size * log2(size * size);
        }

        uint64_t Kernels::bytes_fft(
            uint64_t size,
            uint64_t batch) const
        {
            uint64_t nr_correlations = mConstants.get_nr_correlations();
            return 1ULL * 2 * batch * nr_correlations * size * size * 2 * sizeof(float);
        }

        uint64_t Kernels::flops_adder(
            uint64_t nr_subgrids) const
        {
            uint64_t subgrid_size = mConstants.get_subgrid_size();
            uint64_t nr_correlations = mConstants.get_nr_correlations();
            uint64_t flops = 0;
            flops += 1ULL * nr_subgrids * subgrid_size * subgrid_size * 8; // shift
            flops += 1ULL * nr_subgrids * subgrid_size * subgrid_size * nr_correlations * 2; // add
            return flops;
        }

        uint64_t Kernels::bytes_adder(
            uint64_t nr_subgrids) const
        {
            uint64_t subgrid_size = mConstants.get_subgrid_size();
            uint64_t nr_correlations = mConstants.get_nr_correlations();
            uint64_t bytes = 0;
            bytes += 1ULL * nr_subgrids * 2 * sizeof(int); // coordinate
            bytes += 1ULL * nr_subgrids * subgrid_size * subgrid_size * 2 * sizeof(float); // grid in
            bytes += 1ULL * nr_subgrids * subgrid_size * subgrid_size * 2 * sizeof(float); // subgrid in
            bytes += 1ULL * nr_subgrids * subgrid_size * subgrid_size * 2 * sizeof(float); // subgrid out
            return bytes;
        }

        uint64_t Kernels::flops_splitter(
            uint64_t nr_subgrids) const
        {
            uint64_t subgrid_size = mConstants.get_subgrid_size();
            uint64_t flops = 0;
            flops += 1ULL * nr_subgrids * subgrid_size * subgrid_size * 8; // shift
            return flops;
        }

        uint64_t Kernels::bytes_splitter(
            uint64_t nr_subgrids) const
        {
            uint64_t subgrid_size = mConstants.get_subgrid_size();
            uint64_t nr_correlations = mConstants.get_nr_correlations();
            uint64_t bytes = 0;
            bytes += 1ULL * nr_subgrids * 2 * sizeof(int); // coordinate
            bytes += 1ULL * nr_subgrids * subgrid_size * subgrid_size * 2 * sizeof(float); // grid in
            bytes += 1ULL * nr_subgrids * subgrid_size * subgrid_size * 2 * sizeof(float); // subgrid out
            return bytes;
        }

        uint64_t Kernels::flops_scaler(
            uint64_t nr_subgrids) const
        {
            uint64_t subgrid_size = mConstants.get_subgrid_size();
            uint64_t nr_correlations = mConstants.get_nr_correlations();
            uint64_t flops = 0;
            flops += 1ULL * nr_subgrids * subgrid_size * subgrid_size * nr_correlations * 2; // scale
            return flops;
        }

        uint64_t Kernels::bytes_scaler(
            uint64_t nr_subgrids) const
        {
            uint64_t subgrid_size = mConstants.get_subgrid_size();
            uint64_t nr_correlations = mConstants.get_nr_correlations();
            uint64_t bytes = 0;
            bytes += 1ULL * nr_subgrids * subgrid_size * subgrid_size * nr_correlations * 2 * sizeof(float); // scale
            return bytes;
        }

        template<typename T>
        void Kernels::shift(
            Array3D<T>& data)
        {
            int nr_polarizations = data.get_z_dim();
            int height = data.get_y_dim();
            int width = data.get_x_dim();
            assert(height == width);

            T tmp13, tmp24;

            // Dimensions
            int n = height;
            int n2 = n / 2;

            // Interchange entries in 4 quadrants, 1 <--> 3 and 2 <--> 4
            #pragma omp parallel for private(tmp13, tmp24)
            for (int pol = 0; pol < nr_polarizations; pol++) {
                for (int i = 0; i < n2; i++) {
                    for (int k = 0; k < n2; k++) {
                        tmp13              = x(pol, i, k);
                        x(pol, i, k)       = x(pol, i+n2, k+n2);
                        x(pol, i+n2, k+n2) = tmp13;

                        tmp24              = x(pol, i+n2, k);
                        x(pol, i+n2, k)    = x(pol, i, k+n2);
                        x(pol, i, k+n2)    = tmp24;
                     }
                }
            }
        }

        template<typename T>
        void Kernels::scale(
            Array3D<std::complex<T>>& data,
            std::complex<T> scale)
        {
            int nr_polarizations = data.get_z_dim();
            int height = data.get_y_dim();
            int width = data.get_x_dim();

            #pragma omp parallel for collapse(2)
            for (int pol = 0; pol < nr_polarizations; pol++) {
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        std::complex<T> value = x(pol, y, x);
                        x(pol, y, x) = T(
                            value.real() * scale.real(),
                            value.imag() * scale.imag());
                    }
                }
            }
        }

        uint64_t Kernels::sizeof_grid(
            uint64_t grid_size)
        {
            // TODO: also support double precision
            uint64_t nr_correlations = mConstants.get_nr_correlations();
            return 1ULL * nr_correlations * grid_size * grid_size * sizeof(std::complex<float>);
        }

        uint64_t Kernels::sizeof_visibilities(
            uint64_t nr_baselines,
            uint64_t nr_timesteps,
            uint64_t nr_channels)
        {
            uint64_t nr_correlations = mConstants.get_nr_correlations();
            return 1ULL * nr_baselines * nr_timesteps * nr_channels * nr_correlations * sizeof(Visibility<std::complex<float>>);
        }

        uint64_t Kernels::sizeof_uvw(
            uint64_t nr_baselines,
            uint64_t nr_timesteps)
        {
            return 1ULL * nr_baselines * nr_timesteps * sizeof(UVW);
        }

    } // namespace kernel
} // namespace idg
