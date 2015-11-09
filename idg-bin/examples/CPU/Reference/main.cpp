#include "CPU/Reference/idg.h"

#include "common.h"

void run(
    idg::Parameters params,
    int nr_subgrids,
    void *uvw,
    void *wavenumbers,
    void *visibilities,
    void *spheroidal,
    void *aterm,
    void *metadata,
    void *subgrids,
    void *grid
) {
    // Initialize interface to kernels
    clog << ">>> Initialize proxy" << endl;
    idg::proxy::cpu::Reference xeon(params);
    clog << endl;

    // Run gridder
    clog << ">>> Run gridder" << endl;
    int jobsize_gridder = params.get_job_size_gridder();
    xeon.grid_onto_subgrids(jobsize_gridder, nr_subgrids, 0, uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrids);

    clog << ">> Run adder" << endl;
    int jobsize_adder = params.get_job_size_adder();
    xeon.add_subgrids_to_grid(jobsize_adder, nr_subgrids, metadata, subgrids, grid);

    clog << ">> Run fft" << endl;
    xeon.transform(idg::FourierDomainToImageDomain, grid);

    clog << ">>> Run splitter" << endl;
    int jobsize_splitter = params.get_job_size_splitter();
    xeon.split_grid_into_subgrids(jobsize_splitter, nr_subgrids, metadata, subgrids, grid);

    clog << ">>> Run degridder" << endl;
    int jobsize_degridder = params.get_job_size_degridder();
    xeon.degrid_from_subgrids(jobsize_degridder, nr_subgrids, 0, uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrids);
}
