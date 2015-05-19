#include <stdint.h>
#include <cuda.h>
#include <cudaProfiler.h>

#include "Proxies.h"
#include "Init.h"
#include "Arguments.h"
#include "Memory.h"

#include "CU.h"
#include "CUFFT.h"

/*
    Enable/disable parts of the program
*/
#define GRIDDER		1
#define ADDER		1
#define SPLITTER    1
#define DEGRIDDER	1
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
	options << " -DNR_STATIONS="		<< nr_stations;
	options << " -DNR_TIME="			<< nr_time;
	options << " -DNR_CHANNELS="		<< nr_channels;
	options << " -DW_PLANES="           << w_planes;
	options << " -DGRIDSIZE="			<< gridsize;
	options << " -DSUBGRIDSIZE="		<< subgridsize;
	options << " -DCHUNKSIZE="          << chunksize;
	options << " -DJOBSIZE="            << jobsize;
	options << " -DNR_POLARIZATIONS="	<< nr_polarizations;
	options << " -DNR_BASELINES="		<< nr_baselines;
	return options.str();
}


/*
	Main
*/
int main(int argc, char **argv) {
    // Parameters
    int deviceNumber = 0;
    int nr_streams = 0;
    int nr_stations = 0;
    int nr_time = 0;
    int nr_channels = 0;
    int w_planes = 0;
    int gridsize = 0;
    int subgridsize = 0;
    int chunksize = 0;
    int jobsize = 0;
    get_parameters(argc, argv, &nr_stations, &nr_time, &nr_channels, &w_planes, &gridsize, &subgridsize, &chunksize, &jobsize, &deviceNumber, &nr_streams);
    int nr_polarizations = 4;
    int nr_chunks =  nr_time / chunksize;
    int nr_baselines = (nr_stations * (nr_stations-1)) / 2;
    float imagesize = 0.1;
    
    // Compiler options
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

    // Compute size of data structures
    uint64_t wavenumber_size   = 1ULL * nr_channels * sizeof(float);
    uint64_t aterm_size        = 1ULL * nr_stations * nr_polarizations * subgridsize * subgridsize * 2 * sizeof(float);
    uint64_t spheroidal_size   = 1ULL * subgridsize * subgridsize * sizeof(float);
    uint64_t baseline_size     = 1ULL * nr_baselines * 2 * sizeof(int);
    uint64_t visibilities_size = 1ULL * nr_baselines * nr_time * nr_channels * nr_polarizations * 2 * sizeof(float);
    uint64_t uvw_size          = 1ULL * nr_baselines * nr_time * 3 * sizeof(float);
    uint64_t subgrid_size      = 1ULL * nr_baselines * nr_chunks * nr_polarizations * subgridsize * subgridsize * 2 * sizeof(float);
    uint64_t grid_size         = 1ULL * nr_polarizations * gridsize * gridsize * 2 * sizeof(float);
    
	// Print size of datastructures
	std::clog << "Size of data" << std::endl;
	std::clog << "\tVisibilities:\t" << visibilities_size / 1e9 << " GB" << std::endl;
    std::clog << "\tUVW:\t\t"        << uvw_size          / 1e9 << " GB" << std::endl;
    std::clog << "\tATerm:\t\t"      << aterm_size        / 1e6 << " MB" << std::endl;
    std::clog << "\tSpheroidal:\t"   << spheroidal_size   / 1e3 << " KB" << std::endl;
    std::clog << "\tBaselines:\t"    << baseline_size     / 1e3 << " KB" << std::endl;
    std::clog << "\tSubgrid:\t"      << subgrid_size      / 1e9 << " GB" << std::endl;
    std::clog << "\tGrid:\t\t"       << grid_size         / 1e9 << " GB" << std::endl;
    std::clog << std::endl;

	// Initialize CUDA
    std::clog << ">>> Initialize CUDA" << std::endl;
    cu::init();
    cu::Device device(deviceNumber);
    cu::Context context(device, cudaHostRegisterMapped);
    context.setCurrent();
   
    // Check memory requirements
    uint64_t required_host_memory = visibilities_size + uvw_size + aterm_size + spheroidal_size +
                                  baseline_size + subgrid_size + grid_size;
    uint64_t free_host_memory = free_memory();
    std::clog << "Memory on host (required/available: ";
    std::clog << required_host_memory / 1e9 << " / ";
    std::clog << free_host_memory / 1e9 << " GB" << std::endl;
    if (0.9 * free_host_memory < required_host_memory) {
        std::clog << "Too little host memory available\n" << std::endl;
        exit(EXIT_FAILURE);
    }
    uint64_t required_device_memory = (visibilities_size + uvw_size + subgrid_size) /
                                    (nr_baselines / (double) jobsize) * nr_streams;
    uint64_t free_device_memory = device.free_memory();
    std::clog << "Memory on device (required/available): ";
    std::clog << required_device_memory / 1e9 << " / ";
    std::clog << free_device_memory  / 1e9 << " GB" << std::endl;
    if (0.9 * free_device_memory < required_device_memory) {
        std::clog << "Too little device memory available\n" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::clog << std::endl;
	
    // Allocate datastructures
    std::clog << ">>> Allocate data structures" << std::endl;
    cu::DeviceMemory d_wavenumbers(wavenumber_size);
    cu::DeviceMemory d_aterm(aterm_size);
    cu::DeviceMemory d_spheroidal(spheroidal_size);
    cu::DeviceMemory d_baselines(baseline_size);
    cu::HostMemory   h_visibilities(visibilities_size);
    cu::HostMemory   h_uvw(uvw_size, CU_MEMHOSTALLOC_WRITECOMBINED);
    cu::HostMemory   h_subgrid(subgrid_size);
    cu::HostMemory   h_grid(grid_size);
    
    // Initialize data structures
    std::clog << ">>> Initialize data structures" << std::endl;
    init_visibilities(h_visibilities, nr_baselines, nr_time, nr_channels, nr_polarizations);
	init_uvw(h_uvw, nr_stations, nr_baselines, nr_time, gridsize, subgridsize, w_planes);
	init_subgrid(h_subgrid, nr_baselines, subgridsize, nr_polarizations, nr_chunks);
	void *wavenumbers  = init_wavenumbers(nr_channels);
	void *aterm        = init_aterm(nr_stations, nr_polarizations, subgridsize);
	void *spheroidal   = init_spheroidal(subgridsize);
	void *baselines    = init_baselines(nr_stations, nr_baselines);
	void *grid         = init_grid(gridsize, nr_polarizations);
    d_wavenumbers.set(wavenumbers);
    d_aterm.set(aterm);
    d_spheroidal.set(spheroidal);
    d_baselines.set(baselines);
   
    // Initialize interface to kernels
    std::clog << ">>> Initialize proxies" << std::endl;
	CUDA proxy("g++", "-g", deviceNumber, nr_stations, nr_baselines, nr_time, nr_channels, nr_polarizations, subgridsize, gridsize, imagesize, chunksize);

    // Start profiling
    cuProfilerStart();

	// Run gridder
	#if GRIDDER
	std::clog << ">>> Run gridder" << std::endl;
	proxy.gridder(context, nr_streams, jobsize, h_visibilities, h_uvw, h_subgrid, d_wavenumbers, d_aterm, d_spheroidal, d_baselines);
	#endif
	
	// Run adder
	#if ADDER
	std::clog << ">>> Run adder" << std::endl;
	proxy.adder(context, nr_streams, jobsize, h_subgrid, h_uvw, h_grid);
	#endif
	
	// Run fft
	#if FFT
	std::clog << ">> Run fft" << std::endl;
	proxy.fft(context, h_grid, CUFFT_FORWARD);
	#endif
	
	// Run splitter
	#if SPLITTER
	std::clog << ">> Run splitter" << std::endl;
	proxy.splitter(context, nr_streams, jobsize, h_subgrid, h_uvw, h_grid);
	#endif

	// Run degridder
	#if DEGRIDDER
	std::clog << ">>> Run degridder" << std::endl;
	proxy.degridder(context, nr_streams, jobsize, h_visibilities, h_uvw, h_subgrid, d_wavenumbers, d_spheroidal, d_aterm, d_baselines);
	#endif

    // Stop profiling
    cuProfilerStop();

	// Free memory
	free(wavenumbers);
	free(aterm);
	free(spheroidal);
	free(baselines);

	return EXIT_SUCCESS;
}
