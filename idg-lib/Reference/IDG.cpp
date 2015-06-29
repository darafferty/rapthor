#include <iostream>
#include <sstream>
#include <cstdlib>
#include <string.h>
#include <omp.h>
#include <fftw3.h>
#include <likwid.h>

#include "Util.h"
#include "Init.h"
#include "Power.h"
#include "Write.h"
#include "Types.h"

#include "KernelGridder.cpp"
#include "KernelDegridder.cpp"
#include "KernelFFT.cpp"
#include "KernelAdder.cpp"
#include "KernelSplitter.cpp"


/*
    Enable/disable parts of the program
*/
#define GRIDDER		1
#define ADDER		1
#define SPLITTER    1
#define DEGRIDDER	1
#define FFT         1
#define WRITE		0


/*
    Performance reporting
*/
#define REPORT_VERBOSE    1
#define REPORT_TOTAL      1
#define SCIENTIFIC_OUTPUT 0


/*
    Performance reporting
*/
void report(const char *message, uint64_t flops, uint64_t bytes, const PowerSensor::State &startState, const PowerSensor::State &stopState) {
    #pragma omp critical (clog)
    {
    double runtime = PowerSensor::seconds(startState, stopState);
    double energy  = PowerSensor::Joules(startState, stopState);

    std::clog << message << ": " << runtime << " s, "
              << flops * 1e-9 / runtime  << " GFLOPS, "
              << bytes * 1e-9  / runtime << " GB/s";
    #if MEASURE_POWER
    if (runtime > 1e-2) {
        std::clog << ", " << energy / runtime << " W" 
                     ", " << flops * 1e-9 / energy   << " GFLOPS/W";
    }
    #endif
    std::clog << std::endl;
    }
}


void report(const char *message, double runtime, uint64_t flops, uint64_t bytes) {
	#pragma omp critical (clog)
	{
    std::clog << message << ": " << runtime << " s";
    if (flops != 0)
	std::clog << ", " << flops / runtime * 1e-9 << " GFLOPS";
    if (bytes != 0)
	std::clog << ", " << bytes / runtime * 1e-9 << " GB/s";
    std::clog << std::endl;
	}
}

void report_visibilities(double runtime) {
    int nr_visibilities = NR_BASELINES * NR_TIME * NR_CHANNELS;
    std::clog << "throughput: " << 1e-6 * nr_visibilities / runtime << " Mvisibilities/s" << std::endl;
}

void report_subgrids(double runtime) {
    std::clog << "throughput: " << 1e-3 * NR_BASELINES / runtime << " Ksubgrids/s" << std::endl;
}

/*
	Arithmetic intensity
*/
void kernel_info() {
	std::clog << ">>> Arithmetic intensity" << std::endl;
	std::clog << "  gridder: " << (float) kernel_gridder_flops(1) /
	                                      kernel_gridder_bytes(1) << std::endl;
	std::clog << "degridder: " << (float) kernel_degridder_flops(1) /
	                                      kernel_degridder_bytes(1) << std::endl;
	std::clog << "      fft: " << (float) kernel_fft_flops(SUBGRIDSIZE, 1) /
	                                      kernel_fft_bytes(SUBGRIDSIZE, 1) << std::endl;
	std::clog << "    adder: " << (float) kernel_adder_flops(1) /
	                                      kernel_adder_bytes(1) << std::endl;
	std::clog << " splitter: " << (float) kernel_splitter_flops(1) /
	                                      kernel_splitter_bytes(1) << std::endl;
	std::clog << std::endl;
}


/*
	Gridder
*/
void run_gridder(
	int jobsize,
	void *visibilities, void *uvw, void *wavenumbers,
	void *aterm, void *spheroidal, void *baselines, void *subgrid) {
    // Runtime counters
    double total_runtime_gridder = 0;
    double total_runtime_fft = 0;
    
    // Power states
    PowerSensor::State startState, stopState;
    PowerSensor::State powerStates[4];
	
    // Start gridder
    startState = powerSensor.read();
	for (int bl = 0; bl < NR_BASELINES; bl += jobsize) {
		// Prevent overflow
		jobsize = bl + jobsize > NR_BASELINES ? NR_BASELINES - bl : jobsize;
		
		// Number of elements in batch
		int uvw_elements          = NR_TIME * 3;
		int visibilities_elements = NR_TIME * NR_CHANNELS * NR_POLARIZATIONS;
		int subgrid_elements      = NR_CHUNKS * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS;
		
		// Pointers to data for current batch
        void *uvw_ptr          = (float *) uvw + bl * uvw_elements;
        void *wavenumbers_ptr  = wavenumbers;
		void *visibilities_ptr = (float complex *) visibilities + bl * visibilities_elements;
		void *spheroidal_ptr   = spheroidal;
		void *aterm_ptr        = aterm;
		void *subgrid_ptr      = (float complex *) subgrid + bl * subgrid_elements;
		void *baselines_ptr    = baselines;
		
	    powerStates[0] = powerSensor.read();
		kernel_gridder(
		    jobsize, bl, (UVWType *) uvw_ptr, (WavenumberType *) wavenumbers_ptr, (VisibilitiesType *) visibilities_ptr,
		    (SpheroidalType *) spheroidal_ptr, (ATermType *) aterm_ptr, (BaselineType *) baselines_ptr, (SubGridType *) subgrid_ptr);
	    powerStates[1] = powerSensor.read();
		
		powerStates[2] = powerSensor.read();
		#if ORDER == ORDER_BL_V_U_P
        kernel_fft(SUBGRIDSIZE, jobsize*NR_POLARIZATIONS, (fftwf_complex *) subgrid_ptr, FFTW_BACKWARD, FFT_LAYOUT_YXP);
        #elif ORDER == ORDER_BL_P_V_U
        kernel_fft(SUBGRIDSIZE, jobsize*NR_POLARIZATIONS, (fftwf_complex *) subgrid_ptr, FFTW_BACKWARD, FFT_LAYOUT_PYX);
        #endif
		powerStates[3] = powerSensor.read();
		
        #if REPORT_VERBOSE
        report("gridder", kernel_gridder_flops(jobsize),
                          kernel_gridder_bytes(jobsize),
                          powerStates[0], powerStates[1]);
		report("    fft", kernel_fft_flops(SUBGRIDSIZE, jobsize),
                          kernel_fft_bytes(SUBGRIDSIZE, jobsize),
                          powerStates[2], powerStates[3]);
		#endif
		#if REPORT_TOTAL
		total_runtime_gridder += PowerSensor::seconds(powerStates[0], powerStates[1]);
		total_runtime_fft     += PowerSensor::seconds(powerStates[2], powerStates[3]);
		#endif
	}
    stopState = powerSensor.read();

    #if REPORT_VERBOSE
    std::clog << std::endl;
    #endif
    #if REPORT_TOTAL
    report("gridder", total_runtime_gridder,
                      kernel_gridder_flops(NR_BASELINES),
                      kernel_gridder_bytes(NR_BASELINES));
    report("    fft", total_runtime_fft,
                      kernel_fft_flops(SUBGRIDSIZE, NR_BASELINES),
                      kernel_fft_bytes(SUBGRIDSIZE, NR_BASELINES));
    long total_flops = kernel_gridder_flops(NR_BASELINES) +
                       kernel_fft_flops(SUBGRIDSIZE, NR_BASELINES); 
    long total_bytes = kernel_gridder_bytes(NR_BASELINES) +
                       kernel_fft_bytes(SUBGRIDSIZE, NR_BASELINES); 
    report("  total", total_flops, total_bytes, startState, stopState); 
    report_visibilities(PowerSensor::seconds(startState, stopState));
    std::clog << std::endl;
	#endif
}


/*
	Adder
*/
void run_adder(
	int jobsize,
	void *uvw, void *subgrid, void *grid) {
	// Power states
	PowerSensor::State startState, stopState;
    PowerSensor::State powerStates[2];
	
	// Run adder
	startState = powerSensor.read();
	for (int bl = 0; bl < NR_BASELINES; bl += jobsize) {
		// Prevent overflow
		jobsize = bl + jobsize > NR_BASELINES ? NR_BASELINES - bl : jobsize;
		
		// Number of elements in batch
        int uvw_elements     = NR_TIME * 3;
		int subgrid_elements = NR_CHUNKS * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS;
		
		// Pointer to data for current jobs
		void *uvw_ptr     = (float *) uvw + bl * uvw_elements;
		void *subgrid_ptr = (float complex *) subgrid + bl * subgrid_elements;
		void *grid_ptr    = grid;
	
		powerStates[0] = powerSensor.read();
		kernel_adder(jobsize, (UVWType *) uvw_ptr, (SubGridType *) subgrid_ptr, (GridType *) grid_ptr);
		powerStates[1] = powerSensor.read();

		#if REPORT_VERBOSE
		report("adder", kernel_adder_flops(jobsize),
		                kernel_adder_bytes(jobsize),
		                powerStates[0], powerStates[1]);
		#endif
	}
	stopState = powerSensor.read();
	
    #if REPORT_VERBOSE
    std::clog << std::endl;
    #endif
	
    #if REPORT_TOTAL
    long total_flops = kernel_adder_flops(NR_BASELINES);
    long total_bytes = kernel_adder_bytes(NR_BASELINES);
    report("total", total_flops, total_bytes, startState, stopState); 
    report_subgrids(PowerSensor::seconds(startState, stopState));
    std::clog << std::endl;
	#endif
}


/*
	Splitter
*/
void run_splitter(
	int jobsize,
	void *uvw, void *subgrid, void *grid) {
	// Power states
	PowerSensor::State startState, stopState;
    PowerSensor::State powerStates[2];
	
	// Run splitter
	startState = powerSensor.read();
	for (int bl = 0; bl < NR_BASELINES; bl += jobsize) {
		// Prevent overflow
		jobsize = bl + jobsize > NR_BASELINES ? NR_BASELINES - bl : jobsize;
		
		// Number of elements in batch
        int uvw_elements     = NR_TIME * 3;;
		int subgrid_elements = NR_CHUNKS * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS;
		
		// Pointer to data for current jobs
		void *uvw_ptr     = (float *) uvw + bl * uvw_elements;
		void *subgrid_ptr = (float complex *) subgrid + bl * subgrid_elements;
		void *grid_ptr    = grid;
	
		// Run splitter
		powerStates[0] = powerSensor.read();
		kernel_splitter(jobsize, (UVWType *) uvw_ptr, (SubGridType *) subgrid_ptr, (GridType *) grid_ptr);
		powerStates[1] = powerSensor.read();

		#if REPORT_VERBOSE
		report("splitter", kernel_splitter_flops(jobsize),
		                   kernel_splitter_bytes(jobsize),
		                   powerStates[0], powerStates[1]);
		#endif
	}
    stopState = powerSensor.read();
	
    #if REPORT_VERBOSE
    std::clog << std::endl;
    #endif
	
    #if REPORT_TOTAL
    long total_flops = kernel_splitter_flops(NR_BASELINES);
    long total_bytes = kernel_splitter_bytes(NR_BASELINES);
    report("   total", total_flops, total_bytes, startState, stopState); 
    report_subgrids(PowerSensor::seconds(startState, stopState));
    std::clog << std::endl;
	#endif
}


/*
	Degridder
*/
void run_degridder(
	int jobsize,
	void *wavenumbers, void *aterm, void *baselines,
	void *visibilities, void *uvw, void *spheroidal, void *subgrid) {
    // Zero visibilties
    memset(visibilities, 0, sizeof(VisibilitiesType));
	
    // Runtime counters
    double total_runtime_fft = 0;
    double total_runtime_degridder = 0;
    
    // Power states
    PowerSensor::State startState, stopState;
    PowerSensor::State powerStates[8];

	// Start degridder
	startState = powerSensor.read();
	for (int bl = 0; bl < NR_BASELINES; bl += jobsize) {
		// Prevent overflow
		jobsize = bl + jobsize > NR_BASELINES ? NR_BASELINES - bl : jobsize;
		
		// Number of elements in batch
		int uvw_elements          = NR_TIME * 3;
		int visibilities_elements = NR_TIME * NR_CHANNELS * NR_POLARIZATIONS;
		int subgrid_elements      = NR_CHUNKS * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS;
		
		// Pointers to data for current batch
        void *uvw_ptr          = (float *) uvw + bl * uvw_elements;
        void *wavenumbers_ptr  = wavenumbers;
		void *visibilities_ptr = (float complex *) visibilities + bl * visibilities_elements;
		void *spheroidal_ptr   = spheroidal;
		void *aterm_ptr        = aterm;
		void *subgrid_ptr      = (float complex *) subgrid + bl * subgrid_elements;
		void *baselines_ptr    = baselines;
		
		powerStates[0] = powerSensor.read();
		#if ORDER == ORDER_BL_V_U_P
        kernel_fft(SUBGRIDSIZE, jobsize*NR_POLARIZATIONS, (fftwf_complex *) subgrid_ptr, FFTW_FORWARD, FFT_LAYOUT_YXP);
        #elif ORDER == ORDER_BL_P_V_U
        kernel_fft(SUBGRIDSIZE, jobsize*NR_POLARIZATIONS, (fftwf_complex *) subgrid_ptr, FFTW_FORWARD, FFT_LAYOUT_PYX);
        #endif
        powerStates[1] = powerSensor.read();

		powerStates[2] = powerSensor.read();
		kernel_degridder(
		    jobsize, bl, (SubGridType *) subgrid_ptr, (UVWType *) uvw_ptr, (WavenumberType *) wavenumbers_ptr,
		    (ATermType *) aterm_ptr, (BaselineType *) baselines_ptr, (SpheroidalType *) spheroidal_ptr, (VisibilitiesType *) visibilities_ptr);
		powerStates[3] = powerSensor.read();

		#if REPORT_VERBOSE
		report("      fft", kernel_fft_flops(SUBGRIDSIZE, jobsize),
		                    kernel_fft_bytes(SUBGRIDSIZE, jobsize),
		                    powerStates[0], powerStates[1]);
		report("degridder", kernel_degridder_flops(jobsize),
		                    kernel_degridder_bytes(jobsize),
		                    powerStates[2], powerStates[3]);
	    #endif
        #if REPORT_TOTAL
		total_runtime_fft       += PowerSensor::seconds(powerStates[0], powerStates[1]);
		total_runtime_degridder += PowerSensor::seconds(powerStates[2], powerStates[3]);
		#endif
	}
	stopState = powerSensor.read();
	
    #if REPORT_VERBOSE
    std::clog << std::endl;
    #endif
	
    #if REPORT_TOTAL
    report("      fft", total_runtime_fft,
                        kernel_fft_flops(SUBGRIDSIZE, NR_BASELINES),
                        kernel_fft_bytes(SUBGRIDSIZE, NR_BASELINES));
    report("degridder", total_runtime_degridder,
                        kernel_degridder_flops(NR_BASELINES),
                        kernel_degridder_bytes(NR_BASELINES));
    long total_flops = kernel_degridder_flops(NR_BASELINES) +
                       kernel_fft_flops(SUBGRIDSIZE, NR_BASELINES); 
    long total_bytes = kernel_degridder_bytes(NR_BASELINES) +
                       kernel_fft_bytes(SUBGRIDSIZE, NR_BASELINES); 
    report("    total", total_flops, total_bytes, startState, stopState); 
    report_visibilities(PowerSensor::seconds(startState, stopState));
    std::clog << std::endl;
	#endif
}


/*
    FFT
*/
void run_fft(
	void *grid,
	int sign) {
    // Start fft
	PowerSensor::State powerStates[2];
	powerStates[0] = powerSensor.read();
	kernel_fft(GRIDSIZE, 1, (fftwf_complex *) grid, sign, FFT_LAYOUT_PYX);
	powerStates[1] = powerSensor.read();

    #if REPORT_TOTAL
    report("fft", kernel_fft_flops(GRIDSIZE, 1),
                  kernel_fft_bytes(GRIDSIZE, 1),
                  powerStates[0], powerStates[1]);
    double runtime_fft = PowerSensor::seconds(powerStates[0], powerStates[1]);
    std::clog << std::endl;
    #endif
}


/*
	Main
*/
int main(int argc, char **argv) {
	// Print configuration
    std::clog << ">>> Configuration"  << std::endl;
    std::clog << "\tStations:\t"      << NR_STATIONS      << std::endl;
    std::clog << "\tBaselines:\t"     << NR_BASELINES     << std::endl;
    std::clog << "\tTimesteps:\t"     << NR_TIME          << std::endl;
    std::clog << "\tChannels:\t"      << NR_CHANNELS      << std::endl;
    std::clog << "\tPolarizations:\t" << NR_POLARIZATIONS << std::endl;
    std::clog << "\tW-planes:\t"      << W_PLANES         << std::endl;
    std::clog << "\tGridsize:\t"      << GRIDSIZE         << std::endl;
    std::clog << "\tSubgridsize:\t"   << SUBGRIDSIZE      << std::endl;
    std::clog << "\tChunksize:\t"     << CHUNKSIZE        << std::endl;
    std::clog << "\tChunks:\t\t"      << NR_CHUNKS        << std::endl;
    std::clog << "\tJobsize:\t"       << JOBSIZE          << std::endl;
    std::clog << std::endl;

    // Initialize data structures
    void *visibilities = init_visibilities(NR_BASELINES, NR_TIME, NR_CHANNELS, NR_POLARIZATIONS);
	void *uvw          = init_uvw(NR_STATIONS, NR_BASELINES, NR_TIME, GRIDSIZE, SUBGRIDSIZE, W_PLANES);
	void *wavenumbers  = init_wavenumbers(NR_CHANNELS);
	void *aterm        = init_aterm(NR_STATIONS, NR_POLARIZATIONS, SUBGRIDSIZE);
	void *spheroidal   = init_spheroidal(SUBGRIDSIZE);
	void *baselines    = init_baselines(NR_STATIONS, NR_BASELINES);
	void *subgrid      = init_subgrid(NR_BASELINES, SUBGRIDSIZE, NR_POLARIZATIONS, NR_CHUNKS);
	void *grid         = init_grid(GRIDSIZE, NR_POLARIZATIONS);
	
	// Print size of datastructures
	std::clog << "Size of data" << std::endl;
	std::clog << "\tVisibilities:\t" << sizeof(VisibilitiesType) / 1e9 << " GB" << std::endl;
    std::clog << "\tUVW:\t\t"        << sizeof(UVWType)          / 1e9 << " GB" << std::endl;
    std::clog << "\tATerm:\t\t"      << sizeof(ATermType)        / 1e6 << " MB" << std::endl;
    std::clog << "\tSpheroidal:\t"   << sizeof(SpheroidalType)   / 1e3 << " KB" << std::endl;
    std::clog << "\tBaselines:\t"    << sizeof(BaselineType)     / 1e3 << " KB" << std::endl;
    std::clog << "\tSubgrid:\t"      << sizeof(SubGridType)      / 1e9 << " GB" << std::endl;
    std::clog << "\tGrid:\t\t"       << sizeof(GridType)         / 1e9 << " GB" << std::endl;
    std::clog << std::endl;
	
    // Show kernel information
    kernel_info();
	
	// Set output mode
	#if SCIENTIFIC_OUTPUT
	std::clog << std::scientific;
	#endif

    // Enable likwid markers
    #if USE_LIKWID
    likwid_markerInit();
    #endif
	
	// Run gridder
	#if GRIDDER
	std::clog << ">>> Run gridder" << std::endl;
	run_gridder(JOBSIZE, visibilities, uvw, wavenumbers, aterm, spheroidal, baselines, subgrid);
	#endif
	
	// Run adder
	#if ADDER
	std::clog << ">>> Run adder" << std::endl;
	run_adder(JOBSIZE, uvw, subgrid, grid);
	#endif
	
	// Run fft
	#if FFT
	std::clog << ">> Run fft" << std::endl;
	run_fft(grid, 0);
	#endif
	
	// Run splitter
	#if SPLITTER
	std::clog << ">> Run splitter" << std::endl;
	run_splitter(JOBSIZE, uvw, subgrid, grid);
	#endif

	// Run degridder
	#if DEGRIDDER
	std::clog << ">>> Run degridder" << std::endl;
	run_degridder(JOBSIZE, wavenumbers, aterm, baselines, visibilities, uvw, spheroidal, subgrid);
	#endif

    // Close likwid markers
    #if USE_LIKWID
    likwid_markerClose();
    #endif

    #if WRITE
	// Write grid to file
	std::clog << ">>> Write grid" << std::endl;
    writeGrid(grid, "grid_xeon");
    
    // Write uvgrids to file
    std::clog << ">>> Write subgrid" << std::endl;
    writeSubgrid(subgrid, "subgrid_xeon");

    // Write visibilities to file
    std::clog << ">>> Write visibilities" << std::endl;
    writeVisibilities(visibilities, "visibilities_xeon");
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
