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

    auto size_visibilities = 1ULL * nr_baselines*nr_time*
        nr_channels*nr_polarizations;
    auto size_uvw = 1ULL * nr_baselines*nr_time*3;
    auto size_wavenumbers = 1ULL * nr_channels;
    auto size_aterm = 1ULL * nr_stations*nr_timeslots*
        nr_polarizations*subgridsize*subgridsize;
    auto size_spheroidal = 1ULL * subgridsize*subgridsize;
    auto size_grid = 1ULL * nr_polarizations*gridsize*gridsize;
    auto size_baselines = 1ULL * nr_baselines*2;

    auto visibilities = new std::complex<float>[size_visibilities];
    auto uvw = new float[size_uvw];
    auto wavenumbers = new float[size_wavenumbers];
    auto aterm = new std::complex<float>[size_aterm];
    auto aterm_offsets = new int[nr_timeslots+1];
    auto spheroidal = new float[size_spheroidal];
    //auto grid = new std::complex<float>[size_grid];
    auto baselines = new int[size_baselines];

    //void *h_grid = NULL;
    //cuMemHostAlloc(&h_grid, size_grid * sizeof(std::complex<float>), CU_MEMHOSTALLOC_DEVICEMAP);
    void *grid;
    //cuMemHostAlloc(&grid_ptr, size_grid * sizeof(std::complex<float>), CU_MEMHOSTALLOC_DEVICEMAP);
    std::cout << "sizeof_grid: " << size_grid * sizeof(std::complex<float>) << std::endl;
    printf("test 1: %p\n", grid);
    //cuMemHostAlloc(&grid, size_grid * sizeof(std::complex<float>), 0);
    cuMemHostAlloc(&grid, 10, CU_MEMHOSTALLOC_WRITECOMBINED);
    printf("test 2: %p\n", grid);


    idg::init_visibilities(visibilities, nr_baselines,
                           nr_time,
                           nr_channels, nr_polarizations);
    idg::init_uvw(uvw, nr_stations, nr_baselines, nr_time);
    idg::init_wavenumbers(wavenumbers, nr_channels);
    idg::init_aterm(aterm, nr_stations, nr_timeslots, nr_polarizations,
                    subgridsize);
    idg::init_aterm_offsets(aterm_offsets, nr_timeslots, nr_time);
    idg::init_spheroidal(spheroidal, subgridsize);
    idg::init_grid(grid, gridsize, nr_polarizations);
    //idg::init_grid(h_grid, gridsize, nr_polarizations);
    idg::init_baselines(baselines, nr_stations, nr_baselines);
    std::clog << std::endl;

    //void *h_grid;
    //std::cout << "CU_MEMHOSTALLOC_DEVICEMAP=" << CU_MEMHOSTALLOC_DEVICEMAP << std::endl;
    //std::cout << "CU_MEMHOSTREGISTER_DEVICEMAP=" << CU_MEMHOSTREGISTER_DEVICEMAP << std::endl;
    //cuMemHostAlloc((void **) &grid, size_grid * sizeof(std::complex<float>), CU_MEMHOSTALLOC_DEVICEMAP);
    //void *d_grid;
    //CUdeviceptr d_grid;
    //cuMemHostGetDevicePointer(&d_grid, h_grid, 0);


    // Initialize interface to kernels
    clog << ">>> Initialize proxy" << endl;
    idg::proxy::cuda::Jetson proxy(params, deviceNumber);
    clog << endl;

    // Print all CUDA devices
    clog << ">>> CUDA devices" << endl;
    printDevices(deviceNumber);

    // Start profiling
    cuProfilerStart();

    // Run
    clog << ">>> Run fft" << endl;
    CUdeviceptr d_grid;
    cuMemHostGetDevicePointer(&d_grid, grid, 0);
    //proxy.transform(idg::FourierDomainToImageDomain, grid);
    //proxy.transform(idg::FourierDomainToImageDomain, (std::complex<float>*) h_grid);

    //clog << ">>> Run gridder" << endl;
    //proxy.grid_visibilities(visibilities, uvw, wavenumbers, baselines, grid, w_offset, kernel_size, aterm, aterm_offsets, spheroidal);

    //clog << ">>> Run fft" << endl;
    //proxy.transform(idg::FourierDomainToImageDomain, grid);

    //clog << ">>> Run degridder" << endl;
    //proxy.degrid_visibilities(visibilities, uvw, wavenumbers, baselines, grid, w_offset, kernel_size, aterm, aterm_offsets, spheroidal);

    // Stop profiling
    cuProfilerStop();

    // Free memory for data structures
    delete[] visibilities;
    delete[] uvw;
    delete[] wavenumbers;
    delete[] aterm;
    delete[] aterm_offsets;
    delete[] spheroidal;
    //delete[] grid;
    delete[] baselines;
    //cuMemFreeHost((void *) grid);
}
