#include <complex.h>
#include <stdint.h>
#include <string.h>

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
#define RUN_ADDER		0
#define RUN_SPLITTER    0
#define RUN_FFT         0
#define RUN_DEGRIDDER	0


/*
    Performance reporting
*/
#define REPORT_VERBOSE 1
#define REPORT_TOTAL   1


/*
    Enable/disable kernels
*/
#define GRIDDER   1
#define DEGRIDDER 0
#define ADDER     0
#define SPLITTER  0
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
    OpenCL
*/

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
	options << " -DNR_STATIONS="		<< NR_STATIONS;
	options << " -DNR_BASELINES="		<< NR_BASELINES;
	options << " -DNR_TIME="			<< NR_TIME;
	options << " -DNR_CHANNELS="		<< NR_CHANNELS;
	options << " -DNR_POLARIZATIONS="	<< NR_POLARIZATIONS;
	options << " -DSUBGRIDSIZE="		<< SUBGRIDSIZE;
	options << " -DGRIDSIZE="			<< GRIDSIZE;
	options << " -DCHUNKSIZE="          << CHUNKSIZE;
	options << " -DIMAGESIZE="			<< IMAGESIZE;
    options << " -I.";
	return options.str();
}


/*
    Benchmark
*/
void report(const char *name, double runtime, uint64_t flops, uint64_t bytes) {
	#pragma omp critical(clog)
	{
    std::clog << name << ": " << runtime << " s";
    if (flops != 0)
		std::clog << ", " << flops / runtime * 1e-12 << " TFLOPS";
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
	    program.build(devices, argsPtr);
        std::string msg;
        program.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &msg);
	    #if BUILD_LOG
        std::clog << msg << std::endl;
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

    // Compile kernels
    cl::Program program = compile(SOURCE_GRIDDER, context, device);

    // Get kernels
    KernelGridder kernel_gridder = KernelGridder(program, KERNEL_GRIDDER);
    KernelFFT kernel_fft;
	
    // Timing variables
    double total_time_gridder[nr_streams];
    double total_time_fft[nr_streams];
    double total_time_input[nr_streams];
    double total_time_output[nr_streams];
    double total_bytes_input[nr_streams];
    double total_bytes_output[nr_streams];
    long total_jobs[nr_streams];
    for (int t = 0; t < nr_streams; t++) {
        total_time_gridder[t] = 0;
        total_time_fft[t]     = 0;
        total_time_input[t]   = 0;
        total_time_output[t]  = 0;
        total_bytes_input[t]  = 0;
        total_bytes_output[t] = 0;
        total_jobs[t]         = 0;
    }
    
    // Number of iterations per stream
    int nr_iterations = ((NR_BASELINES / nr_streams) / jobsize) + 1;
	
    // Start gridder
	#pragma omp parallel num_threads(nr_streams)
	{
	    // Initialize
        int current_jobsize = jobsize;
        int thread_num = omp_get_thread_num();
        int iteration = 0;
	    
	    // Private device memory
        cl::Buffer d_visibilities = cl::Buffer(context, CL_MEM_READ_WRITE, VISIBILITIES_SIZE);
        cl::Buffer d_uvw          = cl::Buffer(context, CL_MEM_READ_WRITE, UVW_SIZE);
        cl::Buffer d_subgrid      = cl::Buffer(context, CL_MEM_READ_WRITE, SUBGRID_SIZE);
	    
        for (int bl = thread_num * jobsize; bl < NR_BASELINES; bl += nr_streams * jobsize) {
            // Prevent overflow
		    current_jobsize = bl + jobsize > NR_BASELINES ? NR_BASELINES - bl : jobsize;
        
		    // Number of elements in batch
		    size_t visibilities_offset = bl * NR_TIME * NR_CHANNELS * NR_POLARIZATIONS;
		    size_t uvw_offset          = bl * NR_TIME * 3;
		    size_t subgrid_offset      = bl * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS;
		
	        // Copy input data to device
            queue.enqueueCopyBuffer(h_visibilities, d_visibilities, visibilities_offset, 0, VISIBILITIES_SIZE, NULL, NULL);
            queue.enqueueCopyBuffer(h_uvw, d_uvw, uvw_offset, 0, UVW_SIZE, NULL, NULL);
            queue.enqueueCopyBuffer(h_subgrid, d_subgrid, subgrid_offset, 0, SUBGRID_SIZE, NULL, NULL);

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
            std::clog << "gridding " << bl << " " << current_jobsize << std::endl;
            kernel_gridder.launchAsync(queue, current_jobsize, bl, d_uvw, d_wavenumbers, d_visibilities, d_spheroidal, d_aterm, d_baselines, d_subgrid);
            #endif

            // Launch FFT
            #if FFT
            std::clog << "     fft " << bl << " " << current_jobsize << std::endl;
            kernel_fft.launchAsync(queue, d_subgrid, CLFFT_BACKWARD);
            #endif
	        
            // Copy subgrid to host
            #if OUTPUT
	        //dtohstream.memcpyDtoHAsync(subgrid_ptr, d_subgrid, SUBGRID_SIZE);
            #endif

            // Go to next iteration
            total_jobs[thread_num] += current_jobsize;
            iteration++;

            // Check for errors
            try {
                queue.finish();
            } catch (cl::Error &error) {
                std::cerr << "Error finishing queue: " << error.what() << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        // Sum totals
        #if REPORT_TOTAL
        for (int i = 0; i < nr_iterations; i++) {
            total_time_gridder[thread_num] += 0;
            total_time_fft[thread_num]     += 0;
            total_time_input[thread_num]   += 0;
            total_time_output[thread_num]  += 0;
            total_bytes_input[thread_num]  += 0;
            total_bytes_output[thread_num] += 0;
        }
        #endif
	}

    // Measure total runtime
    #if REPORT_TOTAL
    std::clog << std::endl;
    // Report performance per stream
    for (int t = 0; t < nr_streams; t++) {
        //std::clog << "--- stream " << t << " ---" << std::endl;
        //int jobsize = total_jobs[t];
        //report("gridder", total_time_gridder[t],
        //                  kernel_gridder.flops(jobsize),
        //                  kernel_gridder.bytes(jobsize));
        //report("    fft", total_time_fft[t],
        //                  kernel_fft.flops(SUBGRIDSIZE, jobsize),
        //                  kernel_fft.bytes(SUBGRIDSIZE, jobsize));
        //report("  input", total_time_input[t], 0, total_bytes_input[t]);
        //report(" output", total_time_output[t], 0, total_bytes_output[t]);
	    //std::clog << std::endl;
    }
    
    // Report overall performance
    //std::clog << "--- overall ---" << std::endl;
    //long total_flops = kernel_gridder.flops(NR_BASELINES) +
    //                   kernel_fft.flops(SUBGRIDSIZE, NR_BASELINES);
    //report("     total", records_total[0], records_total[1], total_flops, 0);
    //double total_runtime = runtime(records_total[0], records_total[1]);
    //report_visibilities(total_runtime);
    //std::clog << std::endl;
    #endif

    // Terminate clfft
    clfftTeardown();   
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
    
    // Run Gridder
	#if RUN_GRIDDER
	std::clog << ">>> Run gridder" << std::endl;
    run_gridder(
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
