#include "CUDA/Maxwell/idg.h"

#include "common.h"

void run(
    idg::Parameters params,
    unsigned deviceNumber,
    cu::Context &context,
    int nr_subgrids,
    cu::HostMemory &h_uvw,
    cu::DeviceMemory &d_wavenumbers,
    cu::HostMemory &h_visibilities,
    cu::DeviceMemory &d_spheroidal,
    cu::DeviceMemory &d_aterm,
    cu::HostMemory &h_metadata,
    cu::HostMemory &h_subgrids,
    cu::HostMemory &h_grid
) {
    // Initialize interface to kernels
    clog << ">>> Initialize proxy" << endl;
    idg::proxy::cuda::Maxwell cuda(params, deviceNumber);
    clog << endl;

    // Start profiling
    cuProfilerStart();

    // Run gridder
    clog << ">>> Run gridder" << endl;
    cuda.grid_onto_subgrids(context, nr_subgrids, 0, h_uvw, d_wavenumbers, h_visibilities, d_spheroidal, d_aterm, h_metadata, h_subgrids);

    clog << ">>> Run degridder" << endl;
    cuda.degrid_from_subgrids(context, nr_subgrids, 0, h_uvw, d_wavenumbers, h_visibilities, d_spheroidal, d_aterm, h_metadata, h_subgrids);

    // Stop profiling
    cuProfilerStop();
}
