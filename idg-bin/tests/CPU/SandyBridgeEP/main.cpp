#include "CPU/Reference/idg.h"
#include "CPU/SandyBridgeEP/idg.h"

#include "common.h"

using namespace std;


void run_gridding_test(
    const idg::Parameters& params,
    const int nr_subgrids,
    const float* uvw,
    const float* wavenumbers,
    const complex<float>* visibilities,
    const float* spheroidal,
    const complex<float>* aterm,
    const int* metadata,
    complex<float>* subgrids,
    complex<float>* subgrids_ref,
    complex<float>* grid,
    complex<float>* grid_ref
) {
    // Initialize interface to kernels
    clog << ">>> Initialize proxy" << endl;
    idg::proxy::cpu::SandyBridgeEP sandybridge(params);
    idg::proxy::cpu::Reference reference(params);
    clog << endl;

    // Run gridder
    clog << ">>> Run gridder" << endl;
    sandybridge.grid_onto_subgrids(nr_subgrids, 0,
                                 const_cast<float*>(uvw), 
                                 const_cast<float*>(wavenumbers), 
                                 const_cast<complex<float>*>(visibilities), 
                                 const_cast<float*>(spheroidal), 
                                 const_cast<complex<float>*>(aterm), 
                                 const_cast<int*>(metadata), 
                                 subgrids);

    clog << ">> Run adder" << endl;
    sandybridge.add_subgrids_to_grid(nr_subgrids,
                                   const_cast<int*>(metadata), 
                                   subgrids, grid);

    // Run gridder (reference)
    clog << ">>> Run gridder (reference)" << endl;
    reference.grid_onto_subgrids(nr_subgrids, 0,
                                 const_cast<float*>(uvw), 
                                 const_cast<float*>(wavenumbers), 
                                 const_cast<complex<float>*>(visibilities), 
                                 const_cast<float*>(spheroidal), 
                                 const_cast<complex<float>*>(aterm), 
                                 const_cast<int*>(metadata), 
                                 subgrids_ref);

    clog << ">> Run adder (reference)" << endl;
    reference.add_subgrids_to_grid(nr_subgrids,
                                   const_cast<int*>(metadata), 
                                   subgrids_ref, grid_ref);
}



void run_degridding_test(
    const idg::Parameters& params,
    const int nr_subgrids,
    const float* uvw,
    const float* wavenumbers,
    std::complex<float>* visibilities,
    std::complex<float>* visibilities_ref,
    const float* spheroidal,
    const std::complex<float>* aterm,
    const int* metadata,
    std::complex<float>* subgrids,
    const std::complex<float>* grid)
{
    // Initialize interface to kernels
    clog << ">>> Initialize proxy" << endl;
    idg::proxy::cpu::SandyBridgeEP sandybridge(params);
    idg::proxy::cpu::Reference reference(params);
    clog << endl;

    clog << ">>> Run splitter" << endl;
    sandybridge.split_grid_into_subgrids(nr_subgrids, const_cast<int*>(metadata),
                                       subgrids, const_cast<complex<float>*>(grid));

    clog << ">>> Run degridder" << endl;
    sandybridge.degrid_from_subgrids(nr_subgrids, 0, const_cast<float*>(uvw),
                                   const_cast<float*>(wavenumbers), visibilities,
                                   const_cast<float*>(spheroidal),
                                   const_cast<complex<float>*>(aterm),
                                   const_cast<int*>(metadata),
                                   subgrids);

    clog << ">>> Run splitter" << endl;
    reference.split_grid_into_subgrids(nr_subgrids, const_cast<int*>(metadata),
                                       subgrids, const_cast<complex<float>*>(grid));

    clog << ">>> Run degridder" << endl;
    reference.degrid_from_subgrids(nr_subgrids, 0, const_cast<float*>(uvw),
                                   const_cast<float*>(wavenumbers), visibilities_ref,
                                   const_cast<float*>(spheroidal),
                                   const_cast<complex<float>*>(aterm),
                                   const_cast<int*>(metadata),
                                   subgrids);
}




// void run_gridding_ref(
//     idg::Parameters params,
//     int nr_subgrids,
//     void *uvw,
//     void *wavenumbers,
//     void *visibilities,
//     void *spheroidal,
//     void *aterm,
//     void *metadata,
//     void *subgrids,
//     void *grid
// ) {
//     // Initialize interface to kernels
//     clog << ">>> Initialize proxy" << endl;
//     idg::proxy::cpu::Reference reference(params);
//     clog << endl;

//     // Run gridder
//     clog << ">>> Run gridder" << endl;
//     int jobsize_gridder = params.get_job_size_gridder();
//     reference.grid_onto_subgrids(jobsize_gridder, nr_subgrids, 0, uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrids);
// }





// void run(
//     idg::Parameters params,
//     int nr_subgrids,
//     void *uvw,
//     void *wavenumbers,
//     void *visibilities,
//     void *spheroidal,
//     void *aterm,
//     void *metadata,
//     void *subgrids,
//     void *grid
// ) {
//     // Initialize interface to kernels
//     clog << ">>> Initialize proxy" << endl;
//     idg::proxy::cpu::Reference reference(params);
//     idg::proxy::cpu::SandyBridgeEP sandybridge(params);
//     clog << endl;

//     // Run gridder
//     clog << ">>> Run gridder" << endl;
//     int jobsize_gridder = params.get_job_size_gridder();
//     sandybridge.grid_onto_subgrids(jobsize_gridder, nr_subgrids, 0, uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrids);

//     // clog << ">> Run adder" << endl;
//     // int jobsize_adder = params.get_job_size_adder();
//     // sandybridge.add_subgrids_to_grid(jobsize_adder, nr_subgrids, metadata, subgrids, grid);

//     // clog << ">> Run fft" << endl;
//     // sandybridge.transform(idg::FourierDomainToImageDomain, grid);

//     // clog << ">>> Run splitter" << endl;
//     // int jobsize_splitter = params.get_job_size_splitter();
//     // sandybridge.split_grid_into_subgrids(jobsize_splitter, nr_subgrids, metadata, subgrids, grid);

//     // clog << ">>> Run degridder" << endl;
//     // int jobsize_degridder = params.get_job_size_degridder();
//     // sandybridge.degrid_from_subgrids(jobsize_degridder, nr_subgrids, 0, uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrids);
// }
