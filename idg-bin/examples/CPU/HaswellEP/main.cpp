#include "CPU/HaswellEP/idg.h"

#include "common.h"

void run(
    idg::Parameters params,
    int nr_subgrids,
    float *uvw,
    float *wavenumbers,
    complex<float> *visibilities,
    float *spheroidal,
    complex<float> *aterm,
    int *metadata,
    complex<float> *subgrids,
    complex<float> *grid)
{
    // Initialize interface to kernels
    clog << ">>> Initialize proxy" << endl;
    idg::proxy::cpu::HaswellEP xeon(params);
    clog << endl;

    // Run gridder
    clog << ">>> Run gridder" << endl;
    xeon.grid_onto_subgrids(nr_subgrids, 0, uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrids);

    clog << ">> Run adder" << endl;
    xeon.add_subgrids_to_grid(nr_subgrids, metadata, subgrids, grid);

    clog << ">> Run fft" << endl;
    xeon.transform(idg::FourierDomainToImageDomain, grid);

    clog << ">>> Run splitter" << endl;
    xeon.split_grid_into_subgrids(nr_subgrids, metadata, subgrids, grid);

    clog << ">>> Run degridder" << endl;
    xeon.degrid_from_subgrids(nr_subgrids, 0, uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrids);
}
