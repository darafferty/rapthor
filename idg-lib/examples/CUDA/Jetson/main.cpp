#include "idg-cuda.h"

#include "../common/common.h"

int main(int argc, char **argv) {
    // Set constants explicitly in the parameters parameter
    clog << ">>> Configuration"  << endl;
    idg::Parameters params;
    params.set_from_env();

    // Get device number
    char *cstr_deviceNumber = getenv("CUDA_DEVICE");
    unsigned deviceNumber = cstr_deviceNumber ? atoi (cstr_deviceNumber) : 0;

    // Retrieve constants for memory allocation
    int nr_stations = params.get_nr_stations();
    int nr_baselines = params.get_nr_baselines();
    int nr_time = params.get_nr_time();
    int nr_timeslots = params.get_nr_timeslots();
    int nr_channels = params.get_nr_channels();
    int gridsize = params.get_grid_size();
    int subgridsize = params.get_subgrid_size();
    float imagesize = params.get_imagesize();
    int nr_polarizations = 4;
    int nr_subgrids = nr_baselines * nr_timeslots;

    float w_offset = 0;
    int kernel_size = (subgridsize / 4) + 1;

    // Print configuration
    clog << params;
    clog << endl;

    // Allocate and initialize data structures
    std::clog << ">>> Initialize data structures" << std::endl;

    // Size of datastructures in elements
    auto size_visibilities  = 1ULL * nr_baselines*nr_time* nr_channels*nr_polarizations;
    auto size_uvw           = 1ULL * nr_baselines*nr_time*3;
    auto size_wavenumbers   = 1ULL * nr_channels;
    auto size_aterm         = 1ULL * nr_stations*nr_timeslots*nr_polarizations*subgridsize*subgridsize;
    auto size_aterm_offsets = 1ULL * (nr_timeslots+1);
    auto size_spheroidal    = 1ULL * subgridsize*subgridsize;
    auto size_grid          = 1ULL * nr_polarizations*gridsize*gridsize;
    auto size_baselines     = 1ULL * nr_baselines*2;

    // Size of datastructures in bytes
    auto sizeof_visibilities  = 1ULL * size_visibilities * sizeof(std::complex<float>);
    auto sizeof_uvw           = 1ULL * size_uvw * sizeof(float);
    auto sizeof_wavenumbers   = 1ULL * size_wavenumbers * sizeof(float);
    auto sizeof_aterm         = 1ULL * size_aterm * sizeof(std::complex<float>);
    auto sizeof_aterm_offsets = 1ULL * size_aterm_offsets * sizeof(int);
    auto sizeof_spheroidal    = 1ULL * size_spheroidal * sizeof(float);
    auto sizeof_grid          = 1ULL * size_grid * sizeof(std::complex<float>);
    auto sizeof_baselines     = 1ULL * size_baselines * sizeof(int);

    // Print size of datastructures
    std::clog << ">> Sizeof datastructures" << endl;
    std::clog.precision(3);
    std::clog << "visibilities:  " << std::fixed << sizeof_visibilities  / 1e6 << " Mb" << std::endl;
    std::clog << "uvw:           " << std::fixed << sizeof_uvw           / 1e6 << " Mb" << std::endl;
    std::clog << "wavenumbers:   " << std::fixed << sizeof_wavenumbers   / 1e6 << " Mb" << std::endl;
    std::clog << "aterm:         " << std::fixed << sizeof_aterm         / 1e6 << " Mb" << std::endl;
    std::clog << "aterm offsets: " << std::fixed << sizeof_aterm_offsets / 1e6 << " Mb" << std::endl;
    std::clog << "spheroidal:    " << std::fixed << sizeof_spheroidal    / 1e6 << " Mb" << std::endl;
    std::clog << "grid:          " << std::fixed << sizeof_grid          / 1e6 << " Mb" << std::endl;
    std::clog << "baselines:     " << std::fixed << sizeof_baselines     / 1e6 << " Mb" << std::endl;
    std::clog << std::endl;

    // Initialize interface to kernels
    clog << ">>> Initialize proxy" << endl;
    idg::proxy::cuda::Jetson proxy(params, deviceNumber);
    clog << endl;

    // Allocate CUDA host memory
    clog << ">>> Allocate CUDA host memory" << endl;
    std::complex<float> *visibilities;
    float *uvw;
    float *wavenumbers;
    std::complex<float> *aterm;
    int *aterm_offsets;
    float *spheroidal;
    std::complex<float> *grid;
    int *baselines;
    cuMemHostAlloc((void **) &visibilities, sizeof_visibilities, CU_MEMHOSTREGISTER_DEVICEMAP);
    cuMemHostAlloc((void **) &uvw, sizeof_uvw, CU_MEMHOSTREGISTER_DEVICEMAP);
    cuMemHostAlloc((void **) &wavenumbers, sizeof_wavenumbers, CU_MEMHOSTREGISTER_DEVICEMAP);
    cuMemHostAlloc((void **) &aterm, sizeof_aterm, CU_MEMHOSTREGISTER_DEVICEMAP);
    cuMemHostAlloc((void **) &aterm_offsets, sizeof_aterm_offsets, CU_MEMHOSTREGISTER_DEVICEMAP);
    cuMemHostAlloc((void **) &spheroidal, sizeof_spheroidal, CU_MEMHOSTREGISTER_DEVICEMAP);
    cuMemHostAlloc((void **) &grid, sizeof_grid, CU_MEMHOSTREGISTER_DEVICEMAP);
    cuMemHostAlloc((void **) &baselines, sizeof_baselines, CU_MEMHOSTREGISTER_DEVICEMAP);

    // Initialize data
    clog << ">>> Initialize data" << endl;
    idg::init_visibilities(visibilities, nr_baselines, nr_time, nr_channels, nr_polarizations);
    idg::init_uvw(uvw, nr_stations, nr_baselines, nr_time);
    idg::init_wavenumbers(wavenumbers, nr_channels);
    idg::init_aterm(aterm, nr_stations, nr_timeslots, nr_polarizations, subgridsize);
    idg::init_aterm_offsets(aterm_offsets, nr_timeslots, nr_time);
    idg::init_spheroidal(spheroidal, subgridsize);
    idg::init_grid(grid, gridsize, nr_polarizations);
    idg::init_baselines(baselines, nr_stations, nr_baselines);
    std::clog << std::endl;

    // Start profiling
    cuProfilerStart();

    // Run
    clog << ">>> Run fft" << endl;
    proxy.transform(idg::FourierDomainToImageDomain, grid);

    clog << ">>> Run gridder" << endl;
    proxy.grid_visibilities(visibilities, uvw, wavenumbers, baselines, grid, w_offset, kernel_size, aterm, aterm_offsets, spheroidal);

    clog << ">>> Run fft" << endl;
    proxy.transform(idg::FourierDomainToImageDomain, grid);

    clog << ">>> Run degridder" << endl;
    proxy.degrid_visibilities(visibilities, uvw, wavenumbers, baselines, grid, w_offset, kernel_size, aterm, aterm_offsets, spheroidal);

    // Stop profiling
    cuProfilerStop();
}
