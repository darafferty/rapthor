#include "Proxies.h"
#include "Util.h"
#include "Init.h"

/*
    Enable/disable parts of the program
*/
#define GRIDDER		1
#define ADDER		1
#define DEGRIDDER	1
#define WRITE		0

/*
    Number of uv grids to process at a time
*/
#define JOBSIZE     4096

/*
	Main
*/
int main(int argc, char **argv) {
	// Program parameters
	int deviceNumber = argc >= 2 ? atoi(argv[1]) : 0;
	int nr_streams   = argc >= 3 ? atoi(argv[2]) : 1;
	const char *powerSensor = argc >= 4 ? argv[3] : NULL;
	int jobsize = JOBSIZE;
	
	// Compiler options
	const char *cc = "g++";
	const char *cflags = "-lcuda -lcufft -fopenmp";
	
	// Compile init module
    Init init(cc, "");
        
	// Initialize host datastructures
	std::clog << std::endl << ">>> Initializing datastructures" << std::endl;
	std::clog << "\tVisibilities" << std::endl;
	void *visibilities	= init.init_visibilities();
	std::clog << "\tUVW" << std::endl;
	void *uvw			= init.init_uvw();
	std::clog << "\tOffset" << std::endl;
	void *offset		= init.init_offset();
	std::clog << "\tWavenumbers" << std::endl;
	void *wavenumbers	= init.init_wavenumbers();
	std::clog << "\tATerm" << std::endl;
	void *aterm		    = init.init_aterm();
	std::clog << "\tSpheroidal" << std::endl;
	void *spheroidal	= init.init_spheroidal();
	std::clog << "\tBaselines" << std::endl;
	void *baselines     = init.init_baselines();
	std::clog << "\tCoordinates" << std::endl;
	void *coordinates	= init.init_coordinates();
	std::clog << "\tUVgrid" << std::endl;
	void *uvgrid        = init.init_uvgrid();
	std::clog << "\tGrid" << std::endl << std::endl;
	void *grid          =  init.init_grid();
	
	// Get parameters
	int nr_stations       = init.get_nr_stations();
	int nr_baselines      = init.get_nr_baselines();
	int nr_baselines_data = init.get_nr_baselines_data();
	int nr_time           = init.get_nr_time();
	int nr_time_data      = init.get_nr_time_data();
	int nr_channels       = init.get_nr_channels();
	int nr_polarizations  = init.get_nr_polarizations();
	int blocksize         = init.get_blocksize();
	int gridsize          = init.get_gridsize();
	float imagesize       = init.get_imagesize();
	
	// Print configuration
    std::clog << "Configuration"        << std::endl;
    std::clog << "\tTimesteps:\t"       << nr_time_data      << std::endl;
    std::clog << "\tStations:\t"        << nr_stations       << std::endl;
    std::clog << "\tBaselines:\t"       << nr_baselines_data << std::endl;
    std::clog << "\tChannels:\t"        << nr_channels       << std::endl;
    std::clog << "\tPolarizations:\t"   << nr_polarizations  << std::endl;

    std::clog << "\nProcessing " << nr_baselines << " virtual baselines of "
              << nr_time << " timesteps using " << blocksize << "x" << blocksize
              << " uvgrids" << std::endl << std::endl;
	
	// Initialize interface to kernels
	std::clog << ">>> Initialize proxies" << std::endl;
	bool measure_power = powerSensor != NULL;
	CUDA proxy(cc, cflags, nr_stations, nr_baselines,
	                       nr_time, nr_channels, nr_polarizations,
	                       blocksize, gridsize, imagesize, measure_power);
    proxy.init(deviceNumber, powerSensor);

	// Run Gridder
	#if GRIDDER
	std::clog << ">>> Run gridder" << std::endl;
	proxy.gridder(deviceNumber, nr_streams, jobsize, visibilities, uvw, offset, wavenumbers, aterm, spheroidal, baselines, uvgrid);
	#endif

    // Run adder
    #if ADDER
	std::clog << ">>> Run adder" << std::endl;
	proxy.adder(deviceNumber, nr_streams, jobsize, coordinates, uvgrid, grid);
	#endif

    // Run degridder
    #if DEGRIDDER
	std::clog << ">>> Run degridder" << std::endl;
	proxy.degridder(deviceNumber, nr_streams, jobsize, offset, wavenumbers, aterm, baselines, visibilities, uvw, spheroidal, uvgrid);
	#endif

    #if WRITE
    // Compile util module
    Util util(cc, "", nr_stations, nr_baselines,
                      nr_time, nr_channels, nr_polarizations,
	                  blocksize, gridsize, imagesize);

	// Write grid to file
	std::clog << ">>> Write grid" << std::endl;
    util.writeGrid(grid, "grid_cuda");
    
    // Write uvgrids to file
    std::clog << ">>> Write uvgrids" << std::endl;
    util.writeUVGrid(uvgrid, "uvgrid_cuda");
    
    // Write visibilities to file
    std::clog << ">>> Write visibilities" << std::endl;
    util.writeVisibilities(visibilities, "visibilities_cuda");
    #endif
    
	// Free memory
	free(visibilities);
	free(uvw);
	free(offset);
	free(wavenumbers);
	free(aterm);
	free(spheroidal);
	free(baselines);
	free(coordinates);
	free(uvgrid);
	free(grid);

	return EXIT_SUCCESS;
}
