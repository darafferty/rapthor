#include "Kernels.h"

namespace idg {
namespace kernel {

    /*
        Flop and byte count
    */
    uint64_t flops_gridder(Parameters &parameters, uint64_t nr_timesteps, uint64_t nr_subgrids) {
        uint64_t subgridsize = parameters.get_subgrid_size();
        uint64_t nr_channels = parameters.get_nr_channels();
        uint64_t nr_polarizations = parameters.get_nr_polarizations();

        // Number of flops per visibility
        uint64_t flops_per_visibility = 0;
        flops_per_visibility += 5; // phase index
        flops_per_visibility += 5; // phase offset
        flops_per_visibility += nr_channels * 2; // phase
        #if defined(REPORT_OPS)
        flops_per_visibility += nr_channels * 1; // phasor
        flops_per_visibility += nr_channels * nr_polarizations * 8 / 2; // update
        #else
        flops_per_visibility += nr_channels * nr_polarizations * 8; // update
        #endif

        // Number of flops per subgrid
        uint64_t flops_per_subgrid = 0;
        flops_per_subgrid += nr_polarizations * 30; // aterm
        flops_per_subgrid += nr_polarizations * 2; // spheroidal
        flops_per_subgrid += 6; // shift

        // Total number of flops
        uint64_t flops_total = 0;
        flops_total += nr_timesteps * subgridsize * subgridsize * flops_per_visibility;
        flops_total += nr_subgrids  * subgridsize * subgridsize * flops_per_subgrid;
        return flops_total;
    }

    uint64_t bytes_gridder(Parameters &parameters, uint64_t nr_timesteps, uint64_t nr_subgrids) {
        uint64_t subgridsize = parameters.get_subgrid_size();
        uint64_t nr_channels = parameters.get_nr_channels();
        uint64_t nr_polarizations = parameters.get_nr_polarizations();

        // Number of bytes per uvw coordinate
        uint64_t bytes_per_uvw = 0;
        bytes_per_uvw += 1ULL * 3 * sizeof(float); // read uvw

        // Number of bytes per visibility
        uint64_t bytes_per_vis = 0;
        bytes_per_vis += 1ULL * nr_channels * nr_polarizations * 2 * sizeof(float); // read visibilities

        // Number of bytes per pixel
        uint64_t bytes_per_pix = 0;
        bytes_per_pix += 1ULL * nr_polarizations * 2 * sizeof(float); // read pixel
        bytes_per_pix += 1ULL * nr_polarizations * 2 * sizeof(float); // write pixel

        // Number of bytes per aterm
        uint64_t bytes_per_aterm = 0;
        bytes_per_aterm += 1ULL * 2 * nr_polarizations * 2 * sizeof(float); // read aterm

        // Number of bytes per spheroidal
        uint64_t bytes_per_spheroidal = 0;
        bytes_per_spheroidal += 1ULL * sizeof(float); // read spheroidal

        // Total number of bytes
        uint64_t bytes_total = 0;
        bytes_total += 1ULL * nr_timesteps * bytes_per_uvw;
        bytes_total += 1ULL * nr_timesteps * bytes_per_vis;
        bytes_total += 1ULL * nr_subgrids * subgridsize * subgridsize * bytes_per_pix;
        bytes_total += 1ULL * nr_subgrids * subgridsize * subgridsize * bytes_per_aterm;
        bytes_total += 1ULL * nr_subgrids * subgridsize * subgridsize * bytes_per_spheroidal;
        return bytes_total;
    }

    uint64_t flops_degridder(Parameters &parameters, uint64_t nr_timesteps, uint64_t nr_subgrids) {
        return flops_gridder(parameters, nr_timesteps, nr_subgrids);
    }

    uint64_t bytes_degridder(Parameters &parameters, uint64_t nr_timesteps, uint64_t nr_subgrids) {
        return bytes_gridder(parameters, nr_timesteps, nr_subgrids);
    }

    uint64_t flops_fft(Parameters &parameters, uint64_t size, uint64_t batch) {
        uint64_t nr_polarizations = parameters.get_nr_polarizations();
        // Pseudo number of flops:
        // return 1ULL * 5 * batch * nr_polarizations * size * size * log2(size * size);
        // Estimated number of flops based on fftwf_flops, which seems to
        // return the number of simd instructions, not scalar flops.
        return 1ULL * 4 * batch * nr_polarizations * size * size * log2(size * size);
    }

    uint64_t bytes_fft(Parameters &parameters, uint64_t size, uint64_t batch) {
        uint64_t nr_polarizations = parameters.get_nr_polarizations();
        return 1ULL * 2 * batch * nr_polarizations * size * size * 2 * sizeof(float);
    }

    uint64_t flops_adder(Parameters &parameters, uint64_t nr_subgrids) {
        uint64_t subgridsize = parameters.get_subgrid_size();
        uint64_t nr_polarizations = parameters.get_nr_polarizations();
        uint64_t flops = 0;
        flops += 1ULL * nr_subgrids * subgridsize * subgridsize * 8; // shift
        flops += 1ULL * nr_subgrids * subgridsize * subgridsize * nr_polarizations * 2; // add
        return flops;
    }

    uint64_t bytes_adder(Parameters &parameters, uint64_t nr_subgrids) {
        uint64_t subgridsize = parameters.get_subgrid_size();
        uint64_t nr_polarizations = parameters.get_nr_polarizations();
        uint64_t bytes = 0;
        bytes += 1ULL * nr_subgrids * 2 * sizeof(int); // coordinate
        bytes += 1ULL * nr_subgrids * subgridsize * subgridsize * 2 * sizeof(float); // grid in
        bytes += 1ULL * nr_subgrids * subgridsize * subgridsize * 2 * sizeof(float); // subgrid in
        bytes += 1ULL * nr_subgrids * subgridsize * subgridsize * 2 * sizeof(float); // subgrid out
        return bytes;
    }

    uint64_t flops_splitter(Parameters &parameters, uint64_t nr_subgrids) {
        uint64_t subgridsize = parameters.get_subgrid_size();
        uint64_t flops = 0;
        flops += 1ULL * nr_subgrids * subgridsize * subgridsize * 8; // shift
        return flops;
    }

    uint64_t bytes_splitter(Parameters &parameters, uint64_t nr_subgrids) {
        uint64_t subgridsize = parameters.get_subgrid_size();
        uint64_t nr_polarizations = parameters.get_nr_polarizations();
        uint64_t bytes = 0;
        bytes += 1ULL * nr_subgrids * 2 * sizeof(int); // coordinate
        bytes += 1ULL * nr_subgrids * subgridsize * subgridsize * 2 * sizeof(float); // grid in
        bytes += 1ULL * nr_subgrids * subgridsize * subgridsize * 2 * sizeof(float); // subgrid out
        return bytes;
    }

} // namespace kernel
} // namespace idg
