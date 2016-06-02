#include "Kernels.h"

namespace idg {
namespace kernel {

    /*
        Flop and byte count
    */
    uint64_t flops_gridder(Parameters &parameters, int nr_timesteps, int nr_subgrids) {
        int subgridsize = parameters.get_subgrid_size();
        int nr_channels = parameters.get_nr_channels();
        int nr_polarizations = parameters.get_nr_polarizations();

        // Number of flops per visibility
        uint64_t flops_per_visibility = 0;
        flops_per_visibility += 5; // phase index
        flops_per_visibility += 5; // phase offset
        flops_per_visibility += nr_channels * 2; // phase
        #if defined(COUNT_SINCOS_AS_FLOPS)
        flops_per_visibility += nr_channels * FLOPS_PER_SINCOS; // phasor
        #endif
        flops_per_visibility += nr_channels * nr_polarizations * 8; // update

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

    uint64_t bytes_gridder(Parameters &parameters, int nr_timesteps, int nr_subgrids) {
        int subgridsize = parameters.get_subgrid_size();
        int nr_channels = parameters.get_nr_channels();
        int nr_polarizations = parameters.get_nr_polarizations();

        // Number of bytes per uvw coordinate
        uint64_t bytes_per_uvw = 0;
        bytes_per_uvw += 1ULL * 3 * sizeof(float); // uvw

        // Number of bytes per visibility
        uint64_t bytes_per_vis = 0;
        bytes_per_vis += 1ULL * nr_channels * nr_polarizations * 2 * sizeof(float); // visibilities

        // Number of bytes per pixel
        uint64_t bytes_per_pix = 0;
        bytes_per_pix += 1ULL * nr_polarizations * 2 * sizeof(float); // pixel

        // Total number of bytes
        uint64_t bytes_total = 0;
        bytes_total += 1ULL * nr_timesteps * bytes_per_uvw;
        bytes_total += 1ULL * nr_timesteps * bytes_per_vis;
        bytes_total += 1ULL * nr_subgrids * subgridsize * subgridsize * bytes_per_pix;
        return bytes_total;
    }

    uint64_t flops_degridder(Parameters &parameters, int nr_timesteps, int nr_subgrids) {
        return flops_gridder(parameters, nr_timesteps, nr_subgrids);
    }

    uint64_t bytes_degridder(Parameters &parameters, int nr_timesteps, int nr_subgrids) {
        return bytes_gridder(parameters, nr_timesteps, nr_subgrids);
    }

    uint64_t flops_fft(Parameters &parameters, int size, int batch) {
        int nr_polarizations = parameters.get_nr_polarizations();
        return 1ULL * batch * nr_polarizations * 5 * size * size * log(size * size);
    }

    uint64_t bytes_fft(Parameters &parameters, int size, int batch) {
        int nr_polarizations = parameters.get_nr_polarizations();
        return 1ULL * 2 * batch * nr_polarizations * size * size * 2 * sizeof(float);
    }

    uint64_t flops_adder(Parameters &parameters, int nr_subgrids) {
        int subgridsize = parameters.get_subgrid_size();
        int nr_polarizations = parameters.get_nr_polarizations();
        uint64_t flops = 0;
        flops += 1ULL * nr_subgrids * subgridsize * subgridsize * 8; // shift
        flops += 1ULL * nr_subgrids * subgridsize * subgridsize * nr_polarizations * 2; // add
        return flops;
    }

    uint64_t bytes_adder(Parameters &parameters, int nr_subgrids) {
        int subgridsize = parameters.get_subgrid_size();
        int nr_polarizations = parameters.get_nr_polarizations();
        uint64_t bytes = 0;
        bytes += 1ULL * nr_subgrids * 2 * sizeof(int); // coordinate
        bytes += 1ULL * nr_subgrids * subgridsize * subgridsize * 2 * sizeof(float); // grid in
        bytes += 1ULL * nr_subgrids * subgridsize * subgridsize * 2 * sizeof(float); // subgrid in
        bytes += 1ULL * nr_subgrids * subgridsize * subgridsize * 2 * sizeof(float); // subgrid out
        return bytes;
    }

    uint64_t flops_splitter(Parameters &parameters, int nr_subgrids) {
        int subgridsize = parameters.get_subgrid_size();
        uint64_t flops = 0;
        flops += 1ULL * nr_subgrids * subgridsize * subgridsize * 8; // shift
        return flops;
    }

    uint64_t bytes_splitter(Parameters &parameters, int nr_subgrids) {
        int subgridsize = parameters.get_subgrid_size();
        int nr_polarizations = parameters.get_nr_polarizations();
        uint64_t bytes = 0;
        bytes += 1ULL * nr_subgrids * 2 * sizeof(int); // coordinate
        bytes += 1ULL * nr_subgrids * subgridsize * subgridsize * 2 * sizeof(float); // grid in
        bytes += 1ULL * nr_subgrids * subgridsize * subgridsize * 2 * sizeof(float); // subgrid out
        return bytes;
    }

} // namespace kernel
} // namespace idg
