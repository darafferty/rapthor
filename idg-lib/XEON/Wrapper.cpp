#include <iostream>
#include <sstream>
#include <cstdlib>
#include <string.h>
#include <iomanip>

#include <omp.h>
#include <fftw3.h>

#include <idg/Common/RW.h>
#include <idg/Common/Parameters.h>
#include <idg/Common/Types.h>

#include "Kernels.h"

/*
    Performance reporting
*/
#define REPORT_VERBOSE 0
#define REPORT_TOTAL   1

/*
	File and kernel names
*/
#define SOURCE_GRIDDER      SOURCE_DIR "/XEON/Gridder.cpp"
#define SOURCE_DEGRIDDER    SOURCE_DIR "/XEON/Degridder.cpp"
#define SOURCE_FFT		    SOURCE_DIR "/XEON/FFT.cpp"
#define SOURCE_ADDER	    SOURCE_DIR "/XEON/Adder.cpp"
#define SOURCE_SPLITTER     SOURCE_DIR "/XEON/Splitter.cpp"
#define SOURCE_SHIFTER      SOURCE_DIR "/XEON/Shifter.cpp"
#define SO_GRIDDER          BINARY_DIR "/Gridder.so"
#define SO_DEGRIDDER        BINARY_DIR "/Degridder.so"
#define SO_FFT			    BINARY_DIR "/FFT.so"
#define SO_ADDER		    BINARY_DIR "/Adder.so"
#define SO_SPLITTER         BINARY_DIR "/Splitter.so"
#define SO_SHIFTER          BINARY_DIR "/Shifter.so"
#define KERNEL_GRIDDER      "kernel_gridder"
#define KERNEL_DEGRIDDER    "kernel_degridder"
#define KERNEL_FFT		    "kernel_fft"
#define KERNEL_ADDER	    "kernel_adder"
#define KERNEL_SPLITTER	    "kernel_splitter"
#define KERNEL_SHIFTER      "kernel_shifter"

std::string compileOptions(const char *cflags) {
	std::stringstream options;
    options << " -I " << INCLUDE_DIR;
	options << " -DNR_STATIONS="		<< NR_STATIONS;
	options << " -DNR_BASELINES="		<< NR_BASELINES;
	options << " -DNR_TIME="			<< NR_TIME;
	options << " -DNR_CHANNELS="		<< NR_CHANNELS;
	options << " -DNR_POLARIZATIONS="	<< NR_POLARIZATIONS;
	options << " -DBLOCKSIZE="			<< BLOCKSIZE;
	options << " -DGRIDSIZE="			<< GRIDSIZE;
	options << " -DIMAGESIZE="			<< IMAGESIZE;
    options << " "                      << cflags;
	return options.str();
}

void report(const char *name, double runtime, uint64_t flops, uint64_t bytes) {
	#pragma omp critical (clog)
	{
    std::clog << std::setprecision(3);
    std::clog << std::scientific;
    std::clog << name << ": " << runtime << " s";
    if (flops != 0)
		std::clog << ", " << flops / runtime * 1e-9 << " GFLOPS";
    if (bytes != 0)
		std::clog << ", " << bytes / runtime * 1e-9 << " GB/s";
    std::clog << std::endl;
	}
}

void report_runtime(double runtime) {
    std::clog << std::setprecision(2);
    std::clog << "runtime: " << runtime << " s" << std::endl;
}

void report_visibilities(double runtime) {
    int nr_visibilities = NR_BASELINES * NR_TIME * NR_CHANNELS;
    std::clog << std::setprecision(2);
    std::clog << "throughput: " << 1e-6 * nr_visibilities / runtime << " Mvisibilities/s" << std::endl;
}

void report_subgrids(double runtime) {
    std::clog << std::setprecision(2);
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
	rw::Module module_shifter(SO_SHIFTER);
	
	// Load kernel functions
	KernelGridder kernel_gridder(module_gridder, KERNEL_GRIDDER);
	KernelDegridder kernel_degridder(module_degridder, KERNEL_DEGRIDDER);
	KernelFFT kernel_fft(module_fft, KERNEL_FFT);
	KernelAdder kernel_adder(module_adder, KERNEL_ADDER);
	KernelAdder kernel_splitter(module_splitter, KERNEL_SPLITTER);
	KernelShifter kernel_shifter(module_shifter, KERNEL_SHIFTER);

	std::clog << ">>> Arithmetic intensity" << std::endl;
	std::clog << "  gridder: " << (float) kernel_gridder.flops(1) /
	                                      kernel_gridder.bytes(1) << std::endl;
	std::clog << "degridder: " << (float) kernel_degridder.flops(1) /
	                                      kernel_degridder.bytes(1) << std::endl;
	std::clog << "      fft: " << (float) kernel_fft.flops(BLOCKSIZE, 1) /
	                                      kernel_fft.bytes(BLOCKSIZE, 1) << std::endl;
	std::clog << "    adder: " << (float) kernel_adder.flops(1) /
	                                      kernel_adder.bytes(1) << std::endl;
	std::clog << " splitter: " << (float) kernel_splitter.flops(1) /
	                                      kernel_splitter.bytes(1) << std::endl;
	std::clog << "  shifter: " << (float) kernel_shifter.flops(1) /
	                                      kernel_shifter.bytes(1) << std::endl;
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
            parameters->nr_baselines		!= int(NR_BASELINES)	    ||
			parameters->nr_time				!= int(NR_TIME)			    ||
			parameters->nr_channels			!= int(NR_CHANNELS)		    ||
			parameters->nr_polarizations	!= int(NR_POLARIZATIONS)    ||
			parameters->blocksize			!= int(BLOCKSIZE)	    	||
			parameters->gridsize			!= int(GRIDSIZE)		    ||
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
    #pragma omp section
    compile_kernel(cc, cflags, SOURCE_SHIFTER, SO_SHIFTER, force);
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
	void *visibilities, void *uvw, void *offset, void *wavenumbers,
	void *aterm, void *spheroidal, void *baselines, void *uvgrid) {
	// Load kernel modules
	rw::Module module_gridder(SO_GRIDDER);
	rw::Module module_shifter(SO_SHIFTER);
	rw::Module module_fft(SO_FFT);
	
	// Load kernel functions
	KernelGridder kernel_gridder(module_gridder, KERNEL_GRIDDER);
	KernelShifter kernel_shifter(module_shifter, KERNEL_SHIFTER);
	KernelFFT kernel_fft(module_fft, KERNEL_FFT);

    // Runtime counters
    double total_runtime_gridder = 0;
    double total_runtime_shifter = 0;
    double total_runtime_fft = 0;

	// Start gridder
	double time_start = omp_get_wtime();
	for (int bl = 0; bl < NR_BASELINES; bl += jobsize) {
		// Prevent overflow
		jobsize = bl + jobsize > NR_BASELINES ? NR_BASELINES - bl : jobsize;
		
		// Number of elements in batch
		size_t uvw_elements        = NR_TIME * 3;
		size_t offset_elements     = 3;
		size_t visibility_elements = NR_TIME * NR_CHANNELS * NR_POLARIZATIONS;
		size_t uvgrid_elements     = BLOCKSIZE * BLOCKSIZE * NR_POLARIZATIONS;
		size_t baselines_elements  = 2;
		
		// Pointers to data for current batch
        void *uvw_ptr          = (float *) uvw + bl * uvw_elements;
        void *offset_ptr       = (float *) offset + bl * offset_elements;
        void *wavenumbers_ptr  = wavenumbers;
		void *visibilities_ptr = (float complex *) visibilities + bl * visibility_elements;
		void *spheroidal_ptr   = spheroidal;
		void *aterm_ptr        = aterm;
		void *uvgrid_ptr       = (float complex *) uvgrid + bl * uvgrid_elements;
		void *baselines_ptr    = (int *) baselines + bl * baselines_elements;
		
		double runtime_gridder = omp_get_wtime();
		kernel_gridder.run(
		    jobsize, uvw_ptr, offset_ptr, wavenumbers_ptr, visibilities_ptr,
		    spheroidal_ptr, aterm, baselines_ptr, uvgrid_ptr);
		runtime_gridder = omp_get_wtime() - runtime_gridder;
        total_runtime_gridder += runtime_gridder;
		
		double runtime_shifter1 = omp_get_wtime();
		kernel_shifter.run(jobsize, uvgrid_ptr);
		runtime_shifter1 = omp_get_wtime() - runtime_shifter1;
		total_runtime_shifter += runtime_shifter1;
	
		double runtime_fft = omp_get_wtime();
		kernel_fft.run(BLOCKSIZE, jobsize*NR_POLARIZATIONS, uvgrid_ptr, FFTW_BACKWARD);
		runtime_fft = omp_get_wtime() - runtime_fft;
        total_runtime_fft += runtime_fft;
		
	    double runtime_shifter2 = omp_get_wtime();
		kernel_shifter.run(jobsize, uvgrid_ptr);
		runtime_shifter2 = omp_get_wtime() - runtime_shifter2;
        total_runtime_shifter += runtime_shifter2;
	
        #if REPORT_VERBOSE
		report("gridder", runtime_gridder,
		                  kernel_gridder.flops(jobsize),
		                  kernel_gridder.bytes(jobsize));
		double runtime_shifter = (runtime_shifter1 + runtime_shifter2) / 2;
	    report("shifter", runtime_shifter,
	                      kernel_shifter.flops(jobsize),
	                      kernel_shifter.bytes(jobsize));
		report("    fft", runtime_fft,
		                  kernel_fft.flops(jobsize),
		                  kernel_fft.bytes(jobsize));
		#endif
	}

    #if REPORT_VERBOSE
    std::clog << std::endl;
    #endif

    #if REPORT_TOTAL
    report("gridder", total_runtime_gridder,
                      kernel_gridder.flops(NR_BASELINES),
                      kernel_gridder.bytes(NR_BASELINES));
	report("shifter", total_runtime_shifter/2,
	                  kernel_shifter.flops(NR_BASELINES),
	                  kernel_shifter.bytes(NR_BASELINES));
    report("    fft", total_runtime_fft,
                      kernel_fft.flops(BLOCKSIZE, NR_BASELINES*NR_POLARIZATIONS),
                      kernel_fft.bytes(BLOCKSIZE, NR_BASELINES*NR_POLARIZATIONS));
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
	void *coordinates, void *uvgrid, void *grid) {
	// Load kernel module
	rw::Module module_adder(SO_ADDER);
	
	// Load kernel function
	KernelAdder kernel_adder(module_adder, KERNEL_ADDER);

    // Runtime counter
	double total_runtime_adder = 0;
	
	// Run adder
	double time_start = omp_get_wtime();
	for (int bl = 0; bl < NR_BASELINES; bl += jobsize) {
		// Prevent overflow
		jobsize = bl + jobsize > NR_BASELINES ? NR_BASELINES - bl : jobsize;
		
		// Number of elements in batch
		size_t coordinate_elements = 2;
		size_t uvgrid_elements = BLOCKSIZE * BLOCKSIZE * NR_POLARIZATIONS;
		
		// Pointer to data for current uvgrids
		void *coordinates_ptr = (int *) coordinates + bl * coordinate_elements;
		void *uvgrid_ptr      = (float complex *) uvgrid + bl * uvgrid_elements;
		void *grid_ptr        = grid;
	
		double runtime_adder = omp_get_wtime();
		kernel_adder.run(jobsize, coordinates_ptr, uvgrid_ptr, grid_ptr);
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
	void *coordinates, void *uvgrid, void *grid) {
	// Load kernel module
	rw::Module module_splitter(SO_SPLITTER);
	
	// Load kernel function
	KernelSplitter kernel_splitter(module_splitter, KERNEL_SPLITTER);

    // Runtime counter
	double total_runtime_splitter = 0;
	
	// Run splitter
	double time_start = omp_get_wtime();
	for (int bl = 0; bl < NR_BASELINES; bl += jobsize) {
		// Prevent overflow
		jobsize = bl + jobsize > NR_BASELINES ? NR_BASELINES - bl : jobsize;
		
		// Number of elements in batch
		size_t coordinate_elements = 2;
		size_t uvgrid_elements = BLOCKSIZE * BLOCKSIZE * NR_POLARIZATIONS;
		
		// Pointer to data for current uvgrids
		void *coordinates_ptr = (int *) coordinates + bl * coordinate_elements;
		void *uvgrid_ptr      = (float complex *) uvgrid + bl * uvgrid_elements;
		void *grid_ptr        = grid;
	
		double runtime_splitter = omp_get_wtime();
		kernel_splitter.run(jobsize, coordinates_ptr, uvgrid_ptr, grid_ptr);
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
	void *offset, void *wavenumbers, void *aterm, void *baselines,
	void *visibilities, void *uvw, void *spheroidal, void *uvgrid) {
	// Load kernel modules
	rw::Module module_degridder(SO_DEGRIDDER);
	rw::Module module_fft(SO_FFT);
	rw::Module module_shifter(SO_SHIFTER);
	
	// Load kernel functions
	KernelDegridder kernel_degridder(module_degridder, KERNEL_DEGRIDDER);
	KernelFFT kernel_fft(module_fft, KERNEL_FFT);
	KernelShifter kernel_shifter(module_shifter, KERNEL_SHIFTER);

    // Zero visibilties
    memset(visibilities, 0, sizeof(VisibilitiesType));
	
    // Runtime counters
    double total_runtime_shifter = 0;
    double total_runtime_fft = 0;
    double total_runtime_degridder = 0;

	// Start degridder
	double time_start = omp_get_wtime();
	for (int bl = 0; bl < NR_BASELINES; bl += jobsize) {
		// Prevent overflow
		jobsize = bl + jobsize > NR_BASELINES ? NR_BASELINES - bl : jobsize;
		
		// Number of elements in batch
		size_t uvw_elements        = NR_TIME * 3;
		size_t offset_elements     = 3;
		size_t visibility_elements = NR_TIME * NR_CHANNELS * NR_POLARIZATIONS;
		size_t uvgrid_elements     = BLOCKSIZE * BLOCKSIZE * NR_POLARIZATIONS;
		size_t baselines_elements  = 2;
		
		// Pointers to data for current batch
        void *uvw_ptr          = (float *) uvw + bl * uvw_elements;
        void *offset_ptr       = (float *) offset + bl * offset_elements;
        void *wavenumbers_ptr  = wavenumbers;
		void *visibilities_ptr = (float complex *) visibilities + bl * visibility_elements;
		void *spheroidal_ptr   = spheroidal;
		void *aterm_ptr        = aterm;
		void *uvgrid_ptr       = (float complex *) uvgrid + bl * uvgrid_elements;
		void *baselines_ptr    = (int *) baselines + bl * baselines_elements;
		
        double runtime_shifter1 = omp_get_wtime();
        kernel_shifter.run(jobsize, uvgrid_ptr);
        runtime_shifter1 = omp_get_wtime() - runtime_shifter1;

		double runtime_fft = omp_get_wtime();
		kernel_fft.run(BLOCKSIZE, jobsize*NR_POLARIZATIONS, uvgrid_ptr, FFTW_FORWARD);
		runtime_fft = omp_get_wtime() - runtime_fft;
		total_runtime_fft += runtime_fft;

        double runtime_shifter2 = omp_get_wtime();
        kernel_shifter.run(jobsize, uvgrid_ptr);
        runtime_shifter2 = omp_get_wtime() - runtime_shifter2;
        total_runtime_shifter += runtime_shifter2;

		double runtime_degridder = omp_get_wtime();
		kernel_degridder.run(
		    jobsize, uvgrid_ptr, uvw_ptr, offset_ptr, wavenumbers_ptr,
		    aterm_ptr, baselines_ptr, spheroidal_ptr, visibilities_ptr);
		runtime_degridder = omp_get_wtime() - runtime_degridder;
		total_runtime_degridder += runtime_degridder;
	
		#if REPORT_VERBOSE
		report("      fft", runtime_fft,
		                    kernel_fft.flops(BLOCKSIZE, NR_BASELINES*NR_POLARIZATIONS),
		                    kernel_fft.bytes(BLOCKSIZE, NR_BASELINES*NR_POLARIZATIONS));
		double runtime_shifter = (runtime_shifter1 + runtime_shifter2) / 2;
	    report("  shifter", runtime_shifter,
	                        kernel_shifter.flops(jobsize),
	                        kernel_shifter.bytes(jobsize));
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
                        kernel_fft.flops(BLOCKSIZE, NR_BASELINES*NR_POLARIZATIONS),
                        kernel_fft.bytes(BLOCKSIZE, NR_BASELINES*NR_POLARIZATIONS));
	report("  shifter", total_runtime_shifter/2,
	                    kernel_shifter.flops(NR_BASELINES),
	                    kernel_shifter.bytes(NR_BASELINES));
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
	KernelFFT kernel_fft(module_fft, KERNEL_FFT);

    // Start fft
	double runtime_fft = omp_get_wtime();
	kernel_fft.run(GRIDSIZE, NR_POLARIZATIONS, grid, sign);
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
