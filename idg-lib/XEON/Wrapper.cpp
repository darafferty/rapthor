#include <complex>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <string.h>
#include <iomanip>

#include <omp.h>
#include <fftw3.h>

#include "RW.h"
#include "Kernels.h"
#include "Parameters.h"
#include "Types.h"

/*
    Performance reporting
*/
#define REPORT_VERBOSE 1
#define REPORT_TOTAL   1

/*
	File and kernel names
*/
#define SOURCE_GRIDDER      "KernelGridder.cpp"
#define SOURCE_DEGRIDDER    "KernelDegridder.cpp"
#define SOURCE_FFT		      "KernelFFT.cpp"
#define SOURCE_ADDER	      "KernelAdder.cpp"
#define SOURCE_SPLITTER     "KernelSplitter.cpp"
#define SO_GRIDDER          "Gridder.so"
#define SO_DEGRIDDER        "Degridder.so"
#define SO_FFT			        "FFT.so"
#define SO_ADDER		        "Adder.so"
#define SO_SPLITTER         "Splitter.so"

std::string compileOptions(const char *cflags) {
	std::stringstream options;
	options << " -DNR_STATIONS="		<< NR_STATIONS;
	options << " -DNR_BASELINES="		<< NR_BASELINES;
	options << " -DNR_TIME="			<< NR_TIME;
	options << " -DNR_CHANNELS="		<< NR_CHANNELS;
	options << " -DNR_POLARIZATIONS="	<< NR_POLARIZATIONS;
	options << " -DSUBGRIDSIZE="		<< SUBGRIDSIZE;
	options << " -DGRIDSIZE="			<< GRIDSIZE;
	options << " -DCHUNKSIZE="          << CHUNKSIZE;
    options << " -DNR_CHUNKS="          << NR_CHUNKS;
	options << " -DIMAGESIZE="			<< IMAGESIZE;
    options << " "                      << cflags;
    options << " -I../Common";
	return options.str();
}

void report(const char *name, double runtime, uint64_t flops, uint64_t bytes) {
	#pragma omp critical (clog)
	{
    std::clog << name << ": " << runtime << " s";
    if (flops != 0)
		std::clog << ", " << flops / runtime * 1e-9 << " GFLOPS";
    if (bytes != 0)
		std::clog << ", " << bytes / runtime * 1e-9 << " GB/s";
    std::clog << std::endl;
	}
}

void report_runtime(double runtime) {
    std::clog << "runtime: " << runtime << " s" << std::endl;
}

void report_visibilities(double runtime) {
    int nr_visibilities = NR_BASELINES * NR_TIME * NR_CHANNELS;
    std::clog << "throughput: " << 1e-6 * nr_visibilities / runtime << " Mvisibilities/s" << std::endl;
}

void report_subgrids(double runtime) {
    std::clog << "throughput: " << 1e-3 * NR_BASELINES / runtime << " Ksubgrids/s" << std::endl;
}

extern "C" {

/*
	Misc
*/
void kernel_info() {
	// Load kernel modules
	rw::Module module_gridder(SO_GRIDDER);
	rw::Module module_degridder(SO_DEGRIDDER);
	rw::Module module_fft(SO_FFT);
	rw::Module module_adder(SO_ADDER);
	rw::Module module_splitter(SO_SPLITTER);
	
	// Load kernel functions
	KernelGridder kernel_gridder(module_gridder);
	KernelDegridder kernel_degridder(module_degridder);
	KernelFFT kernel_fft(module_fft);
	KernelAdder kernel_adder(module_adder);
	KernelSplitter kernel_splitter(module_splitter);

	std::clog << ">>> Arithmetic intensity" << std::endl;
	std::clog << "  gridder: " << (float) kernel_gridder.flops(1) /
	                                      kernel_gridder.bytes(1) << std::endl;
	std::clog << "degridder: " << (float) kernel_degridder.flops(1) /
	                                      kernel_degridder.bytes(1) << std::endl;
	std::clog << "      fft: " << (float) kernel_fft.flops(SUBGRIDSIZE, 1) /
	                                      kernel_fft.bytes(SUBGRIDSIZE, 1) << std::endl;
	std::clog << "    adder: " << (float) kernel_adder.flops(1) /
	                                      kernel_adder.bytes(1) << std::endl;
	std::clog << " splitter: " << (float) kernel_splitter.flops(1) /
	                                      kernel_splitter.bytes(1) << std::endl;
	std::clog << std::endl;
}

void compile_kernel(const char *cc, const char *cflags, const char *source, const char *so, bool force = false) {
    // Get compile options
    std::string options = compileOptions(cflags);
	const char *optionsPtr = static_cast<const char *>(options.c_str());

     try {
        if (force) {
            throw std::exception();
        }
     
		// Try to load kernel module
		rw::Module module(so);
		
		// Retrieve parameters from module
		Parameters *parameters = (Parameters *) dlsym(module, "parameters");
		
		// Check if parameters match the arguments
		if (parameters->nr_stations         != int(NR_STATIONS)         ||
			parameters->nr_time				!= int(NR_TIME)			    ||
			parameters->nr_channels			!= int(NR_CHANNELS)		    ||
			parameters->nr_polarizations	!= int(NR_POLARIZATIONS)    ||
			parameters->subgridsize			!= int(SUBGRIDSIZE)	    	||
			parameters->gridsize			!= int(GRIDSIZE)		    ||
			parameters->chunksize           != int(CHUNKSIZE)           ||
			parameters->imagesize			!= float(IMAGESIZE)) {
			throw std::exception();
		}
	} catch (std::exception &exception) {
		// Compile kernel module
		rw::Source(source).compile(cc, so, optionsPtr);
	}
}

void compile(const char *cc, const char *cflags, bool force = false) {
	std::string options = compileOptions(cflags);
	const char *optionsPtr = static_cast<const char *>(options.c_str());
	#pragma omp parallel sections
	{
	#pragma omp section
    compile_kernel(cc, cflags, SOURCE_GRIDDER, SO_GRIDDER, force);
	#pragma omp section
    compile_kernel(cc, cflags, SOURCE_DEGRIDDER, SO_DEGRIDDER, force);
	#pragma omp section
    compile_kernel(cc, cflags, SOURCE_ADDER, SO_ADDER, force);
	#pragma omp section
    compile_kernel(cc, cflags, SOURCE_SPLITTER, SO_SPLITTER, force);
	#pragma omp section
    compile_kernel(cc, cflags, SOURCE_FFT, SO_FFT, force);
	}
	#pragma omp barrier
	std::clog << std::endl;
}

void init(const char *cc, const char *cflags) {
    compile(cc, cflags, true);
    kernel_info();
}


/*
	Gridder
*/
void run_gridder(
	int jobsize,
	void *visibilities, void *uvw, void *wavenumbers,
	void *aterm, void *spheroidal, void *baselines, void *subgrid) {
	// Load kernel modules
	rw::Module module_gridder(SO_GRIDDER);
	rw::Module module_fft(SO_FFT);
	
	// Load kernel functions
	KernelGridder kernel_gridder(module_gridder);
	KernelFFT kernel_fft(module_fft);

    // Runtime counters
    double total_runtime_gridder = 0;
    double total_runtime_fft = 0;

	// Start gridder
	double time_start = omp_get_wtime();
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
		void *visibilities_ptr = (FLOAT_COMPLEX *) visibilities + bl * visibilities_elements;
		void *spheroidal_ptr   = spheroidal;
		void *aterm_ptr        = aterm;
		void *subgrid_ptr      = (FLOAT_COMPLEX *) subgrid + bl * subgrid_elements;
		void *baselines_ptr    = baselines;
		
		double runtime_gridder = omp_get_wtime();
		kernel_gridder.run(
		    jobsize, bl, uvw_ptr, wavenumbers_ptr, visibilities_ptr,
		    spheroidal_ptr, aterm_ptr, baselines_ptr, subgrid_ptr);
		runtime_gridder = omp_get_wtime() - runtime_gridder;
        total_runtime_gridder += runtime_gridder;
		
		double runtime_fft = omp_get_wtime();
		#if ORDER == ORDER_BL_V_U_P
        kernel_fft.run(SUBGRIDSIZE, jobsize, subgrid_ptr, FFTW_BACKWARD, FFT_LAYOUT_YXP);
        #elif ORDER == ORDER_BL_P_V_U
        kernel_fft.run(SUBGRIDSIZE, jobsize, subgrid_ptr, FFTW_BACKWARD, FFT_LAYOUT_PYX);
        #endif

		runtime_fft = omp_get_wtime() - runtime_fft;
        total_runtime_fft += runtime_fft;
		
        #if REPORT_VERBOSE
		report("gridder", runtime_gridder,
		                  kernel_gridder.flops(jobsize),
		                  kernel_gridder.bytes(jobsize));
		report("    fft", runtime_fft,
                          kernel_fft.flops(SUBGRIDSIZE, NR_BASELINES),
                          kernel_fft.bytes(SUBGRIDSIZE, NR_BASELINES));
		#endif
	}

    #if REPORT_VERBOSE
    std::clog << std::endl;
    #endif

    #if REPORT_TOTAL
    report("gridder", total_runtime_gridder,
                      kernel_gridder.flops(NR_BASELINES),
                      kernel_gridder.bytes(NR_BASELINES));
    report("    fft", total_runtime_fft,
                      kernel_fft.flops(SUBGRIDSIZE, NR_BASELINES),
                      kernel_fft.bytes(SUBGRIDSIZE, NR_BASELINES));
	double time_stop = omp_get_wtime();
	double runtime = time_stop - time_start;
    report_runtime(runtime);
    report_visibilities(runtime);
    std::clog << std::endl;
	#endif
}


/*
	Adder
*/
void run_adder(
	int jobsize,
	void *uvw, void *subgrid, void *grid) {
	// Load kernel module
	rw::Module module_adder(SO_ADDER);
	
	// Load kernel function
	KernelAdder kernel_adder(module_adder);

    // Runtime counter
	double total_runtime_adder = 0;
	
	// Run adder
	double time_start = omp_get_wtime();
	for (int bl = 0; bl < NR_BASELINES; bl += jobsize) {
		// Prevent overflow
		jobsize = bl + jobsize > NR_BASELINES ? NR_BASELINES - bl : jobsize;
		
		// Number of elements in batch
        int uvw_elements     = NR_TIME * 3;
		int subgrid_elements = NR_CHUNKS * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS;
		
		// Pointer to data for current jobs
		void *uvw_ptr     = (float *) uvw + bl * uvw_elements;
		void *subgrid_ptr = (FLOAT_COMPLEX *) subgrid + bl * subgrid_elements;
		void *grid_ptr    = grid;
	
		double runtime_adder = omp_get_wtime();
		kernel_adder.run(jobsize, uvw_ptr, subgrid_ptr, grid_ptr);
		runtime_adder = omp_get_wtime() - runtime_adder;
		total_runtime_adder += runtime_adder;

		#if REPORT_VERBOSE
		report("adder", runtime_adder,
		                kernel_adder.flops(jobsize),
		                kernel_adder.bytes(jobsize));
		#endif
	}
	
    #if REPORT_VERBOSE
    std::clog << std::endl;
    #endif
	
    #if REPORT_TOTAL
    report("adder", total_runtime_adder,
                    kernel_adder.flops(NR_BASELINES),
                    kernel_adder.bytes(NR_BASELINES));
	double time_stop = omp_get_wtime();
	double runtime = time_stop - time_start;
    report_runtime(runtime);
    report_subgrids(runtime);
    std::clog << std::endl;
	#endif
}


/*
	Splitter
*/
void run_splitter(
	int jobsize,
	void *uvw, void *subgrid, void *grid) {
	// Load kernel module
	rw::Module module_splitter(SO_SPLITTER);
	
	// Load kernel function
	KernelSplitter kernel_splitter(module_splitter);

    // Runtime counter
	double total_runtime_splitter = 0;
	
	// Run splitter
	double time_start = omp_get_wtime();
	for (int bl = 0; bl < NR_BASELINES; bl += jobsize) {
		// Prevent overflow
		jobsize = bl + jobsize > NR_BASELINES ? NR_BASELINES - bl : jobsize;
		
		// Number of elements in batch
        int uvw_elements     = NR_TIME * 3;;
		int subgrid_elements = NR_CHUNKS * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS;
		
		// Pointer to data for current jobs
		void *uvw_ptr     = (float *) uvw + bl * uvw_elements;
		void *subgrid_ptr = (FLOAT_COMPLEX *) subgrid + bl * subgrid_elements;
		void *grid_ptr    = grid;
	
		double runtime_splitter = omp_get_wtime();
		kernel_splitter.run(jobsize, uvw_ptr, subgrid_ptr, grid_ptr);
		runtime_splitter = omp_get_wtime() - runtime_splitter;
		total_runtime_splitter += runtime_splitter;

		#if REPORT_VERBOSE
		report("splitter", runtime_splitter,
		                   kernel_splitter.flops(jobsize),
		                   kernel_splitter.bytes(jobsize));
		#endif
	}
	
    #if REPORT_VERBOSE
    std::clog << std::endl;
    #endif
	
    #if REPORT_TOTAL
    report("splitter", total_runtime_splitter,
                       kernel_splitter.flops(NR_BASELINES),
                       kernel_splitter.bytes(NR_BASELINES));
	double time_stop = omp_get_wtime();
	double runtime = time_stop - time_start;
    report_runtime(runtime);
    report_subgrids(runtime);
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
	// Load kernel modules
	rw::Module module_degridder(SO_DEGRIDDER);
	rw::Module module_fft(SO_FFT);
	
	// Load kernel functions
	KernelDegridder kernel_degridder(module_degridder);
	KernelFFT kernel_fft(module_fft);

    // Zero visibilties
    memset(visibilities, 0, sizeof(VisibilitiesType));
	
    // Runtime counters
    double total_runtime_fft = 0;
    double total_runtime_degridder = 0;

	// Start degridder
	double time_start = omp_get_wtime();
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
		void *visibilities_ptr = (FLOAT_COMPLEX *) visibilities + bl * visibilities_elements;
		void *spheroidal_ptr   = spheroidal;
		void *aterm_ptr        = aterm;
		void *subgrid_ptr      = (FLOAT_COMPLEX *) subgrid + bl * subgrid_elements;
		void *baselines_ptr    = baselines;
		
		double runtime_fft = omp_get_wtime();
        #if ORDER == ORDER_BL_V_U_P
		kernel_fft.run(SUBGRIDSIZE, jobsize, subgrid_ptr, FFTW_FORWARD, FFT_LAYOUT_YXP);
        #elif ORDER == ORDER_BL_P_V_U
		kernel_fft.run(SUBGRIDSIZE, jobsize, subgrid_ptr, FFTW_FORWARD, FFT_LAYOUT_PYX);
        #endif
		runtime_fft = omp_get_wtime() - runtime_fft;
		total_runtime_fft += runtime_fft;

		double runtime_degridder = omp_get_wtime();
		kernel_degridder.run(
		    jobsize, bl, subgrid_ptr, uvw_ptr, wavenumbers_ptr,
		    aterm_ptr, baselines_ptr, spheroidal_ptr, visibilities_ptr);
        runtime_degridder = omp_get_wtime() - runtime_degridder;
		total_runtime_degridder += runtime_degridder;
	
		#if REPORT_VERBOSE
		report("      fft", runtime_fft,
		                    kernel_fft.flops(SUBGRIDSIZE, NR_BASELINES),
		                    kernel_fft.bytes(SUBGRIDSIZE, NR_BASELINES));
		report("degridder", runtime_degridder,
		                    kernel_degridder.flops(jobsize),
		                    kernel_degridder.bytes(jobsize));
	    #endif
	}
	
    #if REPORT_VERBOSE
    std::clog << std::endl;
    #endif
	
    #if REPORT_TOTAL
    report("      fft", total_runtime_fft,
                        kernel_fft.flops(SUBGRIDSIZE, NR_BASELINES),
                        kernel_fft.bytes(SUBGRIDSIZE, NR_BASELINES));
    report("degridder", total_runtime_degridder,
                        kernel_degridder.flops(NR_BASELINES),
                        kernel_degridder.bytes(NR_BASELINES));
	double time_stop = omp_get_wtime();
	double runtime = time_stop - time_start;
    report_runtime(runtime);
    report_visibilities(runtime);
    std::clog << std::endl;
	#endif
}

void run_fft(
	void *grid,
	int sign) {
	// Load kernel module
	rw::Module module_fft(SO_FFT);
	
	// Load kernel function
	KernelFFT kernel_fft(module_fft);

    // Start fft
	double runtime_fft = omp_get_wtime();
	kernel_fft.run(GRIDSIZE, 1, grid, sign, FFT_LAYOUT_PYX);
	runtime_fft = omp_get_wtime() - runtime_fft;

    #if REPORT_TOTAL
    report("fft", runtime_fft,
                  kernel_fft.flops(GRIDSIZE, 1),
                  kernel_fft.bytes(GRIDSIZE, 1));
    report_runtime(runtime_fft);
    std::clog << std::endl;
    #endif
}

}
