#include "idg-knc.h"

#include "../../CPU/common/common.h"

using namespace std;

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
    complex<float> *grid) {
    // Initialize interface to kernels
    clog << ">>> Initialize proxy" << endl;
    idg::proxy::hybrid::KNC knc(params);
    clog << endl;

    // Run gridder
    clog << ">>> Run gridder" << endl;
    int jobsize_gridder = params.get_job_size_gridder();
    knc.grid_onto_subgrids(nr_subgrids, 0, uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrids);

    clog << ">> Run adder" << endl;
    int jobsize_adder = params.get_job_size_adder();
    knc.add_subgrids_to_grid(nr_subgrids, metadata, subgrids, grid);

    clog << ">> Run fft" << endl;
    knc.transform(idg::FourierDomainToImageDomain, grid);

    clog << ">>> Run splitter" << endl;
    int jobsize_splitter = params.get_job_size_splitter();
    knc.split_grid_into_subgrids(nr_subgrids, metadata, subgrids, grid);

    clog << ">>> Run degridder" << endl;
    int jobsize_degridder = params.get_job_size_degridder();
    knc.degrid_from_subgrids(nr_subgrids, 0, uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrids);
}
