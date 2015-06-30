/// IDG-WRAPPER.cpp: brief description
/**
 * More details
 */

#include "Proxies.h"
#include "Init.h"
// #include "Intel.h"
#include "Arguments.h"

// using namespace std;

/*
    Enable/disable parts of the program
*/
#define GRIDDER     1
#define ADDER       1
#define SPLITTER    1
#define DEGRIDDER   1
#define FFT         1


/*
    Derived parameters
*/
#define NR_CHUNKS NR_TIME / CHUNKSIZE
#define NR_BASELINES (NR_STATIONS * (NR_STATIONS-1)) / 2


std::string compileOptions(
                           int nr_stations, int nr_time, int nr_channels,
                           int w_planes, int gridsize, int subgridsize,
                           int chunksize, int jobsize, int nr_polarizations,
                           int nr_baselines) {
    std::stringstream options;
    options << " -DNR_STATIONS="<< nr_stations;
    options << " -DNR_TIME="<< nr_time;
    options << " -DNR_CHANNELS="<< nr_channels;
    options << " -DW_PLANES="           << w_planes;
    options << " -DGRIDSIZE="<< gridsize;
    options << " -DSUBGRIDSIZE="<< subgridsize;
    options << " -DCHUNKSIZE="          << chunksize;
    options << " -DJOBSIZE="            << jobsize;
    options << " -DNR_POLARIZATIONS="<< nr_polarizations;
    options << " -DNR_BASELINES="<< nr_baselines;
    return options.str();
}


/*
  Main
*/
int main(int argc, char **argv) {
    // Parameters
    int nr_stations = 0;
    int nr_time = 0;
    int nr_channels = 0;
    int w_planes = 0;
    int gridsize = 0;
    int subgridsize = 0;
    int chunksize = 0;
    int jobsize = 0;
    get_parameters(argc, argv, &nr_stations, &nr_time, &nr_channels, &w_planes, &gridsize, &subgridsize, &chunksize, &jobsize);
    int nr_polarizations = 4;
    int nr_chunks =  nr_time / chunksize;
    int nr_baselines = (nr_stations * (nr_stations-1)) / 2;
    float imagesize = 0.1;
    
    // Compiler options
    const char *cc = "icc";  // input parameter
    const char *cflags;      // input parameter
    // if (can_use_intel_core_4th_gen_features()) {
    //     cflags = "-O3 -xCORE-AVX2 -fopenmp -mkl -lmkl_vml_avx2 -lmkl_avx2";
    // } else {
        cflags = "-O3 -xAVX -fopenmp -mkl -lmkl_vml_avx -lmkl_avx";
    // }
    std::string options = compileOptions(nr_stations, nr_time, nr_channels, w_planes, gridsize, subgridsize, chunksize, jobsize, nr_polarizations, nr_baselines);

    // Print configuration
    std::clog << ">>> Configuration"  << std::endl;
    std::clog << "\tStations:\t"      << nr_stations      << std::endl;
    std::clog << "\tBaselines:\t"     << nr_baselines     << std::endl;
    std::clog << "\tTimesteps:\t"     << nr_time          << std::endl;
    std::clog << "\tChannels:\t"      << nr_channels      << std::endl;
    std::clog << "\tPolarizations:\t" << nr_polarizations << std::endl;
    std::clog << "\tW-planes:\t"      << w_planes         << std::endl;
    std::clog << "\tGridsize:\t"      << gridsize         << std::endl;
    std::clog << "\tSubgridsize:\t"   << subgridsize      << std::endl;
    std::clog << "\tChunksize:\t"     << chunksize        << std::endl;
    std::clog << "\tChunks:\t\t"      << nr_chunks        << std::endl;
    std::clog << "\tJobsize:\t"       << jobsize          << std::endl;
    std::clog << std::endl;

    // Initialize data structures
    std::clog << ">>> Initialize data structures" << std::endl;
    void *visibilities = init_visibilities(nr_baselines, nr_time, nr_channels, nr_polarizations);
    void *uvw          = init_uvw(nr_stations, nr_baselines, nr_time, gridsize, subgridsize, w_planes);
    void *wavenumbers  = init_wavenumbers(nr_channels);
    void *aterm        = init_aterm(nr_stations, nr_polarizations, subgridsize);
    void *spheroidal   = init_spheroidal(subgridsize);
    void *baselines    = init_baselines(nr_stations, nr_baselines);
    void *subgrid      = init_subgrid(nr_baselines, subgridsize, nr_polarizations, nr_chunks);
    void *grid         = init_grid(gridsize, nr_polarizations);
    std::clog << std::endl;

    // Initialize interface to kernels
    std::clog << ">> Initialize proxies" << std::endl;
    Xeon proxy(cc, cflags, nr_stations, nr_baselines, nr_time, nr_channels, nr_polarizations, subgridsize, gridsize, imagesize, chunksize);

    // Run gridder
    #if GRIDDER
    std::clog << ">>> Run gridder" << std::endl;
    proxy.gridder(jobsize, visibilities, uvw, wavenumbers, aterm, spheroidal, baselines, subgrid);
    #endif
    
    // Run adder
#if ADDER
    std::clog << ">>> Run adder" << std::endl;
    proxy.adder(jobsize, uvw, subgrid, grid);
    #endif
    
    // Run fft
    #if FFT
    std::clog << ">> Run fft" << std::endl;
    proxy.fft(grid, 0);
    #endif
    
    // Run splitter
    #if SPLITTER
    std::clog << ">> Run splitter" << std::endl;
    proxy.splitter(jobsize, uvw, subgrid, grid);
    #endif

    // Run degridder
    #if DEGRIDDER
    std::clog << ">>> Run degridder" << std::endl;
    proxy.degridder(jobsize, wavenumbers, aterm, baselines, visibilities, uvw, spheroidal, subgrid);
    #endif

    // Free memory
    free(visibilities);
    free(uvw);
    free(wavenumbers);
    free(aterm);
    free(spheroidal);
    free(baselines);
    free(subgrid);
    free(grid);

    return EXIT_SUCCESS;
}
