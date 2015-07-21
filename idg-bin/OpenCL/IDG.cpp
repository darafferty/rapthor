#include <stdint.h>
#include <string.h>

#include <complex>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cstdlib>

#include <omp.h>
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "Util.h"
#include "Init.h"
#include "Memory.h"
#include "Types.h"
#include "Kernels.h"

#include "opencl.h"


/*
    Debugging
*/
#define BUILD_LOG       1


/*
    Enable/disable parts of the program
*/
#define RUN_GRIDDER		1
#define RUN_ADDER		1
#define RUN_SPLITTER    1
#define RUN_FFT         1
#define RUN_DEGRIDDER	1


/*
u   Performance reporting
*/
#define REPORT_VERBOSE 1
#define REPORT_TOTAL   1


/*
    Enable/disable kernels
*/
#define GRIDDER   1
#define DEGRIDDER 1
#define ADDER     1
#define SPLITTER  1
#define FFT       1
#define INPUT     1
#define OUTPUT    1


/*
	File and kernel names
*/
#define SOURCE_GRIDDER      "Gridder.cl"
#define SOURCE_DEGRIDDER    "Degridder.cl"
#define SOURCE_ADDER		"Adder.cl"
#define SOURCE_SPLITTER     "Splitter.cl"
#define KERNEL_DEGRIDDER    "kernel_degridder"
#define KERNEL_GRIDDER      "kernel_gridder"
#define KERNEL_ADDER        "kernel_adder"
#define KERNEL_SPLITTER     "kernel_splitter"


/*
    Size of device datastructures for one block of work
*/
#define VISIBILITIES_SIZE current_jobsize * NR_TIME * NR_CHANNELS * NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX)
#define UVW_SIZE		  current_jobsize * NR_TIME * 3 * sizeof(float)
#define SUBGRID_SIZE	  current_jobsize * NR_CHUNKS * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX)


/*
    Info
*/
void printDevices(int deviceNumber) {
    // Get context
	cl::Context context = cl::Context(CL_DEVICE_TYPE_ALL);

	// Get devices
	std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
	
	std::clog << "Devices" << std::endl;
	for (int d = 0; d < devices.size(); d++) {
		cl::Device device = devices[d];
		device_info_t devInfo = getDeviceInfo(device);    
		std::clog << "Device: "			  << devInfo.deviceName;
		if (d == deviceNumber) {
			std::clog << "\t" << "<---";
		}
		std::clog << std::endl;
		std::clog << "Driver version  : " << devInfo.driverVersion << std::endl;
		std::clog << "Compute units   : " << devInfo.numCUs << std::endl;
		std::clog << "Clock frequency : " << devInfo.maxClockFreq << " MHz" << std::endl;
        std::clog << std::endl;
    }
	std::clog << "\n";
}


/*
    Compilation
*/
std::string compileOptions() {
	std::stringstream options;
	options << " -DNR_STATIONS="	  << NR_STATIONS;
	options << " -DNR_BASELINES="	  << NR_BASELINES;
	options << " -DNR_TIME="		  << NR_TIME;
	options << " -DNR_CHANNELS="	  << NR_CHANNELS;
	options << " -DNR_POLARIZATIONS=" << NR_POLARIZATIONS;
	options << " -DSUBGRIDSIZE="	  << SUBGRIDSIZE;
	options << " -DGRIDSIZE="		  << GRIDSIZE;
	options << " -DCHUNKSIZE="        << CHUNKSIZE;
	options << " -DIMAGESIZE="		  << IMAGESIZE;
    options << " -DJOBSIZE="          << JOBSIZE;
    options << " -DORDER="            << ORDER;
    options << " -I.";
	return options.str();
}


/*
    Throughput
*/
void report_visibilities(double runtime) {
    int nr_visibilities = NR_BASELINES * NR_TIME * NR_CHANNELS;
    std::clog << "    throughput: ";
    std::clog << 1e-6 * nr_visibilities / runtime << " Mvisibilities/s" << std::endl;
}


void report_subgrids(double runtime) {
    std::clog << "    throughput: ";
    std::clog << 1e-3 * NR_BASELINES / runtime << " Ksubgrids/s" << std::endl;
}


/*
    Compilation
*/
cl::Program compile(const char *filename, cl::Context context, cl::Device device) {
    // Open source file
	std::ifstream source_file(filename);
	std::string source(std::istreambuf_iterator<char>(source_file),
					  (std::istreambuf_iterator<char>()));
	source_file.close();

    // Get arguments
	std::string args = compileOptions();
	const char *argsPtr = static_cast<const char *>(args.c_str());

    // Create vector of devices
    std::vector<cl::Device> devices;
    devices.push_back(device);

    // Try to build the program
    cl::Program program(context, source);
    try {
        #if BUILD_LOG
        std::clog << "Compiling " << filename << ":" << std::endl << args << std::endl;
        #endif
	    program.build(devices, argsPtr);
        std::string msg;
        program.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &msg);
	    #if BUILD_LOG
        std::clog << msg;
	    #endif
    } catch (cl::Error &error) {
        if (strcmp(error.what(), "clBuildProgram") == 0) {
            std::string msg;
            program.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &msg);
            std::cerr << msg << std::endl;
            exit(EXIT_FAILURE);
        }
    }
	
	return program;
}

/*
	Gridder
*/
void run_gridder(
    cl::Context &context, cl::Device &device, cl::CommandQueue &queue,
    int nr_streams, int jobsize,
    cl::Buffer &h_visibilities, cl::Buffer &h_uvw,
    cl::Buffer &h_subgrid, cl::Buffer &d_wavenumbers,
    cl::Buffer &d_spheroidal, cl::Buffer &d_aterm,
    cl::Buffer &d_baselines) {
  
    // Performance counters for io 
    PerformanceCounter counter_input_visibilities("input vis");
    PerformanceCounter counter_input_uvw("input uvw");
    PerformanceCounter counter_output_subgrid("output subgrid");
    PerformanceCounter counter_gridder("gridder");
    PerformanceCounter counter_fft("fft");

    // Compile kernels
    cl::Program program = compile(SOURCE_GRIDDER, context, device);

    // Get kernels
    KernelGridder kernel_gridder = KernelGridder(program, KERNEL_GRIDDER, counter_gridder);
    KernelFFT kernel_fft = KernelFFT(counter_fft);
 
    // Start gridder
    double time_start = omp_get_wtime();
    std::clog << "--- jobs ---" << std::endl;
	#pragma omp parallel num_threads(nr_streams)
	{
	    // Initialize
        int current_jobsize = jobsize;
        int thread_num = omp_get_thread_num();
	    
	    // Private device memory
        cl::Buffer d_visibilities = cl::Buffer(context, CL_MEM_READ_ONLY,  VISIBILITIES_SIZE);
        cl::Buffer d_uvw          = cl::Buffer(context, CL_MEM_READ_ONLY,  UVW_SIZE);
        cl::Buffer d_subgrid      = cl::Buffer(context, CL_MEM_WRITE_ONLY, SUBGRID_SIZE);
	    
        // Events for io
        cl::Event events[3];

        for (int bl = thread_num * jobsize; bl < NR_BASELINES; bl += nr_streams * jobsize)  {
            // Prevent overflow
		    current_jobsize = bl + jobsize > NR_BASELINES ? NR_BASELINES - bl : jobsize;
        
		    // Number of elements in batch
		    size_t visibilities_offset = bl * NR_TIME * NR_CHANNELS * NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX);
		    size_t uvw_offset          = bl * NR_TIME * sizeof(UVW);
		    size_t subgrid_offset      = bl * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX);
            
	        // Copy input data to device
            #if INPUT
            queue.enqueueCopyBuffer(h_uvw, d_uvw, uvw_offset, 0, UVW_SIZE, NULL, &events[0]);
            counter_input_uvw.doOperation(events[0], 0, UVW_SIZE);
            queue.enqueueCopyBuffer(h_visibilities, d_visibilities, visibilities_offset, 0, VISIBILITIES_SIZE, NULL, &events[1]);
            counter_input_visibilities.doOperation(events[1], 0, VISIBILITIES_SIZE);
            #endif

            // Create FFT plan
            #if FFT
            #if ORDER == ORDER_BL_V_U_P
            kernel_fft.plan(context, SUBGRIDSIZE, current_jobsize, FFT_LAYOUT_YXP);
            #elif ORDER == ORDER_BL_P_V_U
            kernel_fft.plan(context, SUBGRIDSIZE, current_jobsize, FFT_LAYOUT_PYX);
            #endif
            #endif

            // Launch gridder kernel
            #if GRIDDER
            kernel_gridder.launchAsync(queue, current_jobsize, bl, d_uvw, d_wavenumbers, d_visibilities, d_spheroidal, d_aterm, d_baselines, d_subgrid);
            #endif

            // Launch FFT
            #if FFT
            kernel_fft.launchAsync(queue, d_subgrid, CLFFT_BACKWARD);
            #endif
	        
            // Copy subgrid to host
            #if OUTPUT
            queue.enqueueCopyBuffer(d_subgrid, h_subgrid, 0, subgrid_offset, SUBGRID_SIZE, NULL, &events[2]);
            counter_output_subgrid.doOperation(events[2], 0, SUBGRID_SIZE);
            #endif
        }

        // Wait for final memory transfer
        #pragma omp barrier
        counter_output_subgrid.wait(); 
	}
    
    // Report overall performance
    double time_end = omp_get_wtime();
    std::clog << std::endl;
    std::clog << "--- overall ---" << std::endl;
    counter_input_uvw.report_total();
    counter_input_visibilities.report_total();
    counter_gridder.report_total();
    counter_fft.report_total();
    counter_output_subgrid.report_total();
    double total_runtime = time_end - time_start;
    PerformanceCounter::report("total", total_runtime, 0, 0);
    report_visibilities(total_runtime);
    std::clog << std::endl;

    // Terminate clfft
    clfftTeardown();   
}

/*
	Degridder
*/
void run_degridder(
    cl::Context &context, cl::Device &device, cl::CommandQueue &queue,
    int nr_streams, int jobsize,
    cl::Buffer &h_visibilities, cl::Buffer &h_uvw,
    cl::Buffer &h_subgrid, cl::Buffer &d_wavenumbers,
    cl::Buffer &d_spheroidal, cl::Buffer &d_aterm,
    cl::Buffer &d_baselines) {

    // Performance counters for io 
    PerformanceCounter counter_input_uvw("input uvw");
    PerformanceCounter counter_input_subgrid("input subgrid");
    PerformanceCounter counter_output_visibilities("output vis");
    PerformanceCounter counter_fft("fft");
    PerformanceCounter counter_degridder("degridder");

    // Compile kernels
    cl::Program program = compile(SOURCE_DEGRIDDER, context, device);

    // Get kernels
    KernelDegridder kernel_degridder = KernelDegridder(program, KERNEL_DEGRIDDER, counter_degridder);
    KernelFFT kernel_fft = KernelFFT(counter_fft);

     // Start degridder
    double time_start = omp_get_wtime();
    std::clog << "--- jobs ---" << std::endl;
	#pragma omp parallel num_threads(nr_streams)
	{
	    // Initialize
        int current_jobsize = jobsize;
        int thread_num = omp_get_thread_num();
	    
	    // Private device memory
        cl::Buffer d_visibilities = cl::Buffer(context, CL_MEM_WRITE_ONLY, VISIBILITIES_SIZE);
        cl::Buffer d_uvw          = cl::Buffer(context, CL_MEM_READ_ONLY,  UVW_SIZE);
        cl::Buffer d_subgrid      = cl::Buffer(context, CL_MEM_READ_ONLY,  SUBGRID_SIZE);
	    
        // Events for io
        cl::Event events[3];

        for (int bl = thread_num * jobsize; bl < NR_BASELINES; bl += nr_streams * jobsize)  {
            // Prevent overflow
		    current_jobsize = bl + jobsize > NR_BASELINES ? NR_BASELINES - bl : jobsize;
        
		    // Number of elements in batch
		    size_t visibilities_offset = bl * NR_TIME * NR_CHANNELS * NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX);
		    size_t uvw_offset          = bl * NR_TIME * sizeof(UVW);
		    size_t subgrid_offset      = bl * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX);
            
	        // Copy input data to device
            #if INPUT
            queue.enqueueCopyBuffer(h_uvw, d_uvw, uvw_offset, 0, UVW_SIZE, NULL, &events[0]);
            counter_input_uvw.doOperation(events[0], 0, UVW_SIZE);
            queue.enqueueCopyBuffer(h_subgrid, d_subgrid, subgrid_offset, 0, SUBGRID_SIZE, NULL, &events[1]);
            counter_input_subgrid.doOperation(events[1], 0, SUBGRID_SIZE);
            #endif

            // Create FFT plan
            #if FFT
            #if ORDER == ORDER_BL_V_U_P
            kernel_fft.plan(context, SUBGRIDSIZE, current_jobsize, FFT_LAYOUT_YXP);
            #elif ORDER == ORDER_BL_P_V_U
            kernel_fft.plan(context, SUBGRIDSIZE, current_jobsize, FFT_LAYOUT_PYX);
            #endif
            #endif

            // Launch FFT
            #if FFT
            kernel_fft.launchAsync(queue, d_subgrid, CLFFT_FORWARD);
            #endif

            // Launch degridder kernel
            #if DEGRIDDER
            kernel_degridder.launchAsync(queue, current_jobsize, bl, d_uvw, d_wavenumbers, d_visibilities, d_spheroidal, d_aterm, d_baselines, d_subgrid);
            #endif

            // Copy visibilities to host
            #if OUTPUT
            queue.enqueueCopyBuffer(d_visibilities, h_visibilities, 0, visibilities_offset, VISIBILITIES_SIZE, NULL, &events[2]);
            counter_output_visibilities.doOperation(events[2], 0, VISIBILITIES_SIZE);
            #endif
        }

        // Wait for all reports
        counter_input_uvw.wait();
        counter_input_subgrid.wait();
        counter_fft.wait();
        counter_degridder.wait();
        counter_output_visibilities.wait();
        #pragma omp barrier
	}

    // Report overall performance
    double time_end = omp_get_wtime();
    std::clog << std::endl;
    std::clog << "--- overall ---" << std::endl;
    double total_runtime = time_end - time_start;
    counter_input_uvw.report_total();
    counter_input_subgrid.report_total();
    counter_fft.report_total();
    counter_degridder.report_total();
    counter_output_visibilities.report_total();
    PerformanceCounter::report("total", total_runtime, 0, 0);
    report_visibilities(total_runtime);
    std::clog << std::endl;

    // Terminate clfft
    clfftTeardown();   
}

/*
    Adder
*/
void run_adder(
    cl::Context &context, cl::Device &device, cl::CommandQueue &queue,
    int nr_streams, int jobsize,
    cl::Buffer &h_subgrid, cl::Buffer &h_uvw,
    cl::Buffer &h_grid) {
  
    // Performance counters for io 
    PerformanceCounter counter_input_uvw("input uvw");
    PerformanceCounter counter_input_subgrid("input subgrid");
    PerformanceCounter counter_output_grid("output grid");
    PerformanceCounter counter_adder("adder");

    // Compile kernels
    cl::Program program = compile(SOURCE_ADDER, context, device);

    // Get kernels
    KernelAdder kernel_adder = KernelAdder(program, KERNEL_ADDER, counter_adder);
 
    // Allocate device memory for grid
    cl::Buffer d_grid(context, CL_MEM_READ_WRITE, sizeof(GridType));

    // Start adder
    double time_start = omp_get_wtime();
    std::clog << "--- jobs ---" << std::endl;
	#pragma omp parallel num_threads(nr_streams)
	{
	    // Initialize
        int current_jobsize = jobsize;
        int thread_num = omp_get_thread_num();
	    
	    // Private device memory
        cl::Buffer d_uvw     = cl::Buffer(context, CL_MEM_READ_ONLY,  UVW_SIZE);
        cl::Buffer d_subgrid = cl::Buffer(context, CL_MEM_WRITE_ONLY, SUBGRID_SIZE);
	    
        // Events for io
        cl::Event events[3];

        for (int bl = thread_num * jobsize; bl < NR_BASELINES; bl += nr_streams * jobsize)  {
            // Prevent overflow
		    current_jobsize = bl + jobsize > NR_BASELINES ? NR_BASELINES - bl : jobsize;
        
		    // Number of elements in batch
		    size_t uvw_offset     = bl * NR_TIME * sizeof(UVW);
		    size_t subgrid_offset = bl * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX);
            
	        // Copy input data to device
            #if INPUT
            queue.enqueueCopyBuffer(h_uvw, d_uvw, uvw_offset, 0, UVW_SIZE, NULL, &events[0]);
            counter_input_uvw.doOperation(events[0], 0, UVW_SIZE);
            queue.enqueueCopyBuffer(h_subgrid, d_subgrid, subgrid_offset, 0, SUBGRID_SIZE, NULL, &events[1]);
            counter_input_subgrid.doOperation(events[1], 0, SUBGRID_SIZE);
            #endif

            // Launch adder kernel
            #if ADDER
            kernel_adder.launchAsync(queue, current_jobsize, bl, d_uvw, d_subgrid, d_grid);
            #endif
        }

        // Copy grid to host
        #if OUTPUT
        queue.enqueueCopyBuffer(d_grid, h_grid, 0, 0, sizeof(GridType), NULL, &events[2]);
        counter_output_grid.doOperation(events[2], 0, sizeof(GridType));
        #endif
        #pragma omp barrier
        counter_output_grid.wait();
	}

    // Report overall performance
    double time_end = omp_get_wtime();
    std::clog << std::endl;
    std::clog << "--- overall ---" << std::endl;
    counter_input_uvw.report_total();
    counter_input_subgrid.report_total();
    counter_adder.report_total();
    counter_output_grid.report_total();
    double total_runtime = time_end - time_start;
    PerformanceCounter::report("total", total_runtime, 0, 0);
    report_subgrids(total_runtime);
    std::clog << std::endl;
}
 
/*
    Splitter
*/
void run_splitter(
    cl::Context &context, cl::Device &device, cl::CommandQueue &queue,
    int nr_streams, int jobsize,
    cl::Buffer &h_subgrid, cl::Buffer &h_uvw,
    cl::Buffer &h_grid) {

    // Performance counters
    PerformanceCounter counter_input_grid("input grid");
    PerformanceCounter counter_input_uvw("input uvw");
    PerformanceCounter counter_output_subgrid("output subgrid");
    PerformanceCounter counter_splitter("splitter");

    // Compile kernels
    cl::Program program = compile(SOURCE_SPLITTER, context, device);

    // Get kernels
    KernelSplitter kernel_splitter = KernelSplitter(program, KERNEL_SPLITTER, counter_splitter);

    // Copy grid to device
    cl::Buffer d_grid(context, CL_MEM_READ_WRITE, sizeof(GridType));
    cl::Event event;
    queue.enqueueCopyBuffer(h_grid, d_grid, 0, 0, sizeof(GridType), NULL, &event);
    counter_input_grid.doOperation(event, 0, sizeof(GridType));

    // Start splitter
    double time_start = omp_get_wtime();
    std::clog << "--- jobs ---" << std::endl;

	#pragma omp parallel num_threads(nr_streams)
	{
	    // Initialize
        int current_jobsize = jobsize;
        int thread_num = omp_get_thread_num();
	    
	    // Private device memory
        cl::Buffer d_uvw     = cl::Buffer(context, CL_MEM_READ_ONLY,  UVW_SIZE);
        cl::Buffer d_subgrid = cl::Buffer(context, CL_MEM_WRITE_ONLY, SUBGRID_SIZE);
	    
        // Events for io
        cl::Event events[2];

        for (int bl = thread_num * jobsize; bl < NR_BASELINES; bl += nr_streams * jobsize)  {
            // Prevent overflow
		    current_jobsize = bl + jobsize > NR_BASELINES ? NR_BASELINES - bl : jobsize;
        
		    // Number of elements in batch
		    size_t uvw_offset     = bl * NR_TIME * sizeof(UVW);
		    size_t subgrid_offset = bl * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * sizeof(FLOAT_COMPLEX);
            
	        // Copy input data to device
            #if INPUT
            queue.enqueueCopyBuffer(h_uvw, d_uvw, uvw_offset, 0, UVW_SIZE, NULL, &events[0]);
            counter_input_uvw.doOperation(events[0], 0, UVW_SIZE);
            #endif

            // Launch splitter kernel
            #if SPLITTER 
            kernel_splitter.launchAsync(queue, current_jobsize, bl, d_uvw, d_subgrid, d_grid);
            #endif

            // Copy subgrid to host
            #if OUTPUT
            queue.enqueueCopyBuffer(d_subgrid, h_subgrid, 0, subgrid_offset, SUBGRID_SIZE, NULL, &events[1]);
            counter_output_subgrid.doOperation(events[1], 0, SUBGRID_SIZE);
            #endif
        }

        #pragma omp barrier
        queue.finish();
        counter_output_subgrid.wait();
	}

    // Report overall performance
    double time_end = omp_get_wtime();
    std::clog << std::endl;
    std::clog << "--- overall ---" << std::endl;
    counter_input_grid.report_total();
    counter_input_uvw.report_total();
    counter_splitter.report_total();
    counter_output_subgrid.report_total();
    double total_runtime = time_end - time_start;
    PerformanceCounter::report("total", total_runtime, 0, 0);
    report_subgrids(total_runtime);
    std::clog << std::endl;
}

/*
	Main
*/
int main(int argc, char **argv) {
	// Program parameters
	int deviceNumber = argc >= 2 ? atoi(argv[1]) : 0;
	int nr_streams   = argc >= 3 ? atoi(argv[2]) : 1;
	int jobsize = JOBSIZE;
	
	/// Print configuration
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

	// Initialize OpenCL
    std::clog << ">>> Initialize OpenCL" << std::endl;
    cl::Context context = cl::Context(CL_DEVICE_TYPE_ALL);
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    cl::Device device = devices[deviceNumber];
    cl::CommandQueue queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
    
    // Show OpenCL devices
	printDevices(deviceNumber);
    
    // Check memory requirements
    // (double the amount of memory is needed for some data structures due to host allocated memory)
    uint64_t required_host_memory = ( 1ULL * 
        2 * sizeof(VisibilitiesType) + 2 * sizeof(UVWType) + sizeof(ATermType) + sizeof(SpheroidalType) +
        sizeof(BaselineType) + 2 * sizeof(SubGridType) + 2 * sizeof(GridType));
    uint64_t free_host_memory = free_memory();
    std::clog << "Memory on host (required/available: ";
    std::clog << required_host_memory / 1e9 << " / ";
    std::clog << free_host_memory / 1e9 << " GB" << std::endl;
    if (0.9 * free_host_memory < required_host_memory) {
        std::clog << "Too little host memory available\n" << std::endl;
        exit(EXIT_FAILURE);
    }
    uint64_t required_device_memory = (
        sizeof(VisibilitiesType) + sizeof(UVWType) + sizeof(SubGridType)) /
        (NR_BASELINES / (double) JOBSIZE) * nr_streams;
    std::clog << "Memory on device (required): ";
    std::clog << required_device_memory / 1e9 << " GB" << std::endl;
    std::clog << std::endl;
	
    // Set output mode
    std::clog << std::setprecision(4);
	
    // Initialize data structures
    std::clog << ">>> Initialize data structures" << std::endl;
    void *visibilities = init_visibilities(NR_BASELINES, NR_TIME, NR_CHANNELS, NR_POLARIZATIONS);
    void *uvw          = init_uvw(NR_STATIONS, NR_BASELINES, NR_TIME, GRIDSIZE, SUBGRIDSIZE, W_PLANES);
    void *subgrid      = init_subgrid(NR_BASELINES, SUBGRIDSIZE, NR_POLARIZATIONS, NR_CHUNKS);
	void *wavenumbers  = init_wavenumbers(NR_CHANNELS);
	void *aterm        = init_aterm(NR_STATIONS, NR_POLARIZATIONS, SUBGRIDSIZE);
	void *spheroidal   = init_spheroidal(SUBGRIDSIZE);
	void *baselines    = init_baselines(NR_STATIONS, NR_BASELINES);
	void *grid         = init_grid(GRIDSIZE, NR_POLARIZATIONS);

    // Initialize OpenCL buffers
    std::clog << ">>> Initialize OpenCL buffers" << std::endl;
    cl::Buffer d_wavenumbers  = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(WavenumberType),       NULL);
    cl::Buffer d_aterm        = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(ATermType),            NULL);
    cl::Buffer d_spheroidal   = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(SpheroidalType),       NULL);
    cl::Buffer d_baselines    = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(BaselineType),         NULL);
    cl::Buffer h_visibilities = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(VisibilitiesType), NULL);
    cl::Buffer h_uvw          = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(UVWType),          NULL);
    cl::Buffer h_subgrid      = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(SubGridType),      NULL);
    cl::Buffer h_grid         = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(GridType),         NULL);
    queue.enqueueWriteBuffer(d_wavenumbers,  CL_TRUE, 0, sizeof(WavenumberType),   wavenumbers,  0, NULL);
    queue.enqueueWriteBuffer(d_aterm,        CL_TRUE, 0, sizeof(ATermType),        aterm,        0, NULL);
    queue.enqueueWriteBuffer(d_spheroidal,   CL_TRUE, 0, sizeof(SpheroidalType),   spheroidal,   0, NULL);
    queue.enqueueWriteBuffer(d_baselines,    CL_TRUE, 0, sizeof(BaselineType),     baselines,    0, NULL);
    queue.enqueueWriteBuffer(h_visibilities, CL_TRUE, 0, sizeof(VisibilitiesType), visibilities, 0, NULL);
    queue.enqueueWriteBuffer(h_uvw,          CL_TRUE, 0, sizeof(UVWType),          uvw,          0, NULL);
    queue.enqueueWriteBuffer(h_subgrid,      CL_TRUE, 0, sizeof(SubGridType),      subgrid,      0, NULL);
    queue.enqueueWriteBuffer(h_grid,         CL_TRUE, 0, sizeof(GridType),         grid,         0, NULL);
    
    // Run gridder
	#if RUN_GRIDDER
	std::clog << ">>> Run gridder" << std::endl;
    run_gridder(
	    context, device, queue, nr_streams, jobsize, h_visibilities, h_uvw, h_subgrid,
	    d_wavenumbers, d_spheroidal, d_aterm, d_baselines);
	#endif

    // Run adder
    #if RUN_ADDER
	std::clog << ">>> Run adder" << std::endl;
    run_adder(
        context, device, queue, nr_streams, jobsize, h_subgrid, h_uvw, h_grid
    );
    #endif

    // Run splitter
    #if RUN_SPLITTER
	std::clog << ">>> Run splitter" << std::endl;
    run_splitter(
        context, device, queue, nr_streams, jobsize, h_subgrid, h_uvw, h_grid
    );
    #endif


    // Run degridder
    #if RUN_DEGRIDDER
	std::clog << ">>> Run degridder" << std::endl;
	run_degridder(
	    context, device, queue, nr_streams, jobsize, h_visibilities, h_uvw, h_subgrid,
	    d_wavenumbers, d_spheroidal, d_aterm, d_baselines);
	#endif

	// Free memory
	free(wavenumbers);
	free(aterm);
	free(spheroidal);
	free(baselines);
	free(grid);

	return EXIT_SUCCESS;
}
