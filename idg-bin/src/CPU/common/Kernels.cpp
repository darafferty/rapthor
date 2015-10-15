#include <cstdint> // unint64_t

#include "idg-config.h"
#include "Kernels.h"

namespace idg {
    namespace kernel {

    // Gridder class
    Gridder::Gridder(runtime::Module &module, Parameters &parameters) :
        _run(module, name_gridder.c_str()),
        parameters(parameters) {}

    void Gridder::run(
            int jobsize, float w_offset, void *uvw, void *wavenumbers,
            void *visibilities, void *spheroidal, void *aterm,
            void *metadata, void *subgrid) {
          (sig_gridder (void *) _run)(jobsize, w_offset, uvw, wavenumbers,
          visibilities, spheroidal, aterm, metadata, subgrid);
    }

    uint64_t Gridder::flops(int jobsize) {
        int subgridsize = parameters.get_subgrid_size();
        int nr_timesteps = parameters.get_nr_timesteps();
        int nr_channels = parameters.get_nr_channels();
        int nr_polarizations = parameters.get_nr_polarizations();
        uint64_t flops = 0;
        flops += 1ULL * jobsize * nr_timesteps * subgridsize * subgridsize * 5; // phase index
        flops += 1ULL * jobsize * nr_timesteps * subgridsize * subgridsize * 5; // phase offset
        flops += 1ULL * jobsize * nr_timesteps * subgridsize * subgridsize * nr_channels * 2; // phase
        flops += 1ULL * jobsize * nr_timesteps * subgridsize * subgridsize * nr_channels * (nr_polarizations * 8); // update
        flops += 1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 30; // aterm
        flops += 1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 2; // spheroidal
        flops += 1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 6; // shift
        return flops;
    }

    uint64_t Gridder::bytes(int jobsize) {
        int subgridsize = parameters.get_subgrid_size();
        int nr_timesteps = parameters.get_nr_timesteps();
        int nr_channels = parameters.get_nr_channels();
        int nr_polarizations = parameters.get_nr_polarizations();
        uint64_t bytes = 0;
        bytes += 1ULL * jobsize * nr_timesteps * 3 * sizeof(float); // uvw
        bytes += 1ULL * jobsize * nr_timesteps * nr_channels * nr_polarizations * 2 * sizeof(float); // visibilities
        bytes += 1ULL * jobsize * nr_polarizations * subgridsize * subgridsize  * 2 * sizeof(float); // subgrids
        return bytes;
    }


    // Degridder class
    Degridder::Degridder(runtime::Module &module, Parameters &parameters) :
        _run(module, name_degridder.c_str()),
        parameters(parameters) {}

    void Degridder::run(
            int jobsize, float w_offset, void *uvw, void *wavenumbers,
            void *visibilities, void *spheroidal, void *aterm,
            void *metadata, void *subgrid) {
          (sig_degridder (void *) _run)(jobsize, w_offset, uvw, wavenumbers,
          visibilities, spheroidal, aterm, metadata, subgrid);
    }

    uint64_t Degridder::flops(int jobsize) {
        int subgridsize = parameters.get_subgrid_size();
        int nr_timesteps = parameters.get_nr_timesteps();
        int nr_channels = parameters.get_nr_channels();
        int nr_polarizations = parameters.get_nr_polarizations();
        uint64_t flops = 0;
        flops += 1ULL * jobsize * nr_timesteps * subgridsize * subgridsize * 5; // phase index
        flops += 1ULL * jobsize * nr_timesteps * subgridsize * subgridsize * 5; // phase offset
        flops += 1ULL * jobsize * nr_timesteps * subgridsize * subgridsize * nr_channels * 2; // phase
        flops += 1ULL * jobsize * nr_timesteps * subgridsize * subgridsize * nr_channels * (nr_polarizations * 8); // update
        flops += 1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 30; // aterm
        flops += 1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 2; // spheroidal
        flops += 1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 6; // shift
        return flops;
    }

    uint64_t Degridder::bytes(int jobsize) {
        int subgridsize = parameters.get_subgrid_size();
        int nr_timesteps = parameters.get_nr_timesteps();
        int nr_channels = parameters.get_nr_channels();
        int nr_polarizations = parameters.get_nr_polarizations();
        uint64_t bytes = 0;
        bytes += 1ULL * jobsize * nr_timesteps * 3 * sizeof(float); // uvw
        bytes += 1ULL * jobsize * nr_timesteps * nr_channels * nr_polarizations * 2 * sizeof(float); // visibilities
        bytes += 1ULL * jobsize * nr_polarizations * subgridsize * subgridsize  * 2 * sizeof(float); // subgrids
        return bytes;
    }


    // GridFFT class
    GridFFT::GridFFT(runtime::Module &module, Parameters &parameters) :
        _run(module, name_fft.c_str()),
        parameters(parameters) {}

    void GridFFT::run(int size, int batch, void *data, int direction) {
        (sig_fft (void *) _run)(size, batch, data, direction);
    }

    uint64_t GridFFT::flops(int size, int batch) {
        int nr_polarizations = parameters.get_nr_polarizations();
	    return 1ULL * batch * nr_polarizations * 5 * size * size * log(size * size);
    }

    uint64_t GridFFT::bytes(int size, int batch) {
        int nr_polarizations = parameters.get_nr_polarizations();
	    return 1ULL * 2 * batch * nr_polarizations * size * size * 2 * sizeof(float);
    }


    // Adder class
    Adder::Adder(runtime::Module &module, Parameters &parameters) :
        _run(module, name_adder.c_str()),
        parameters(parameters) {}

    void Adder::run(int jobsize, void *metadata, void *subgrid, void *grid) {
        (sig_adder (void *) _run)(jobsize, metadata, subgrid, grid);
    }

    uint64_t Adder::flops(int jobsize) {
        int subgridsize = parameters.get_subgrid_size();
        int nr_polarizations = parameters.get_nr_polarizations();
        uint64_t flops = 0;
        flops += 1ULL * jobsize * subgridsize * subgridsize * 8; // shift
        flops += 1ULL * jobsize * subgridsize * subgridsize * 4; // add
        return flops;
    }

    uint64_t Adder::bytes(int jobsize) {
        int subgridsize = parameters.get_subgrid_size();
        int nr_polarizations = parameters.get_nr_polarizations();
        uint64_t bytes = 0;
        bytes += 1ULL * jobsize * subgridsize * subgridsize * 2 * sizeof(unsigned); // coordinate
        bytes += 1ULL * jobsize * subgridsize * subgridsize * 3 * sizeof(unsigned); // pixels
        return bytes;
    }


    // Splitter class
    Splitter::Splitter(runtime::Module &module, Parameters &parameters) :
        _run(module, name_splitter.c_str()),
        parameters(parameters) {}

    void Splitter::run(int jobsize, void *metadata, void *subgrid, void *grid) {
        (sig_splitter (void *) _run)(jobsize, metadata, subgrid, grid);
    }

    uint64_t Splitter::flops(int jobsize) {
        int subgridsize = parameters.get_subgrid_size();
        int nr_polarizations = parameters.get_nr_polarizations();
        uint64_t flops = 0;
        flops += 1ULL * jobsize * subgridsize * subgridsize * 8; // shift
        return flops;
    }

    uint64_t Splitter::bytes(int jobsize) {
        int subgridsize = parameters.get_subgrid_size();
        int nr_polarizations = parameters.get_nr_polarizations();
        uint64_t bytes = 0;
        bytes += 1ULL * jobsize * subgridsize * subgridsize * 2 * sizeof(unsigned); // coordinate
        bytes += 1ULL * jobsize * subgridsize * subgridsize * 3 * sizeof(unsigned); // pixels
        return bytes;
    }

    } // namespace kernel
} // namespace idg
