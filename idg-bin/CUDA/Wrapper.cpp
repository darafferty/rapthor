#include <complex.h>
#include <stdint.h>
#include <string.h>

#include <sstream>
#include <iomanip>

#include <omp.h>
#include <cuda.h>
#include <cufft.h>

#include "Kernels.h"
#include "Power.h"


/*
    Performance reporting
*/
#define REPORT_VERBOSE 0
#define REPORT_TOTAL   1

/*
    Enable/disable kernels
*/
#define GRIDDER   1
#define DEGRIDDER 1
#define ADDER     1
#define SHIFTER   1
#define FFT       1
#define INPUT     1
#define OUTPUT    1
#define WARMUP    1
#define LOOP      0
#define REPEAT    0
#define NR_REPETITIONS 10

/*
	File and kernel names
*/
#define SOURCE_GRIDDER      "CUDA/Gridder.cu"
#define SOURCE_DEGRIDDER    "CUDA/Degridder.cu"
#define SOURCE_ADDER		"CUDA/Adder.cu"
#define SOURCE_SHIFTER      "CUDA/Shifter.cu"
#define PTX_DEGRIDDER       "CUDA/Degridder.ptx"
#define PTX_GRIDDER         "CUDA/Gridder.ptx"
#define PTX_ADDER 			"CUDA/Adder.ptx"
#define PTX_SHIFTER         "CUDA/Shifter.ptx"
#define KERNEL_DEGRIDDER    "kernel_degridder"
#define KERNEL_GRIDDER      "kernel_gridder"
#define KERNEL_ADDER        "kernel_adder"
#define KERNEL_SHIFTER      "kernel_shifter"


/*
    Size of device datastructures for one block of work
*/
#define VISIBILITY_SIZE	current_jobsize * NR_TIME * NR_CHANNELS * NR_POLARIZATIONS * sizeof(float2)
#define UVW_SIZE		current_jobsize * NR_TIME * 3 * sizeof(float)
#define UVGRID_SIZE		current_jobsize * BLOCKSIZE * BLOCKSIZE * NR_POLARIZATIONS * sizeof(float2)

/*
    Info
*/
void printDevices(int deviceNumber) {
	std::clog << "Devices";
	for (int device = 0; device < cu::Device::getCount(); device++) {
		std::clog << "\t" << device << ": ";
		std::clog << cu::Device(device).getName();
		if (device == deviceNumber) {
			std::clog << "\t" << "<---";
		}
		std::clog << std::endl;
	}
	std::clog << "\n";
}


/*
    Compilation
*/
std::string compileOptions(int deviceNumber) {
	std::stringstream options;
	cu::Device device(deviceNumber);
	options << " -DNR_STATIONS="		<< NR_STATIONS;
	options << " -DNR_BASELINES="		<< NR_BASELINES;
	options << " -DNR_TIME="			<< NR_TIME;
	options << " -DNR_CHANNELS="		<< NR_CHANNELS;
	options << " -DNR_POLARIZATIONS="	<< NR_POLARIZATIONS;
	options << " -DBLOCKSIZE="			<< BLOCKSIZE;
	options << " -DGRIDSIZE="			<< GRIDSIZE;
	options << " -DIMAGESIZE="			<< IMAGESIZE;
	int capability = 10 * device.getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR>() +
						  device.getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR>();
	options << " -arch=compute_" << capability;
	options << " -code=sm_" << capability;
	options << " -use_fast_math";
    options << " -lineinfo";
    options << " -src-in-ptx";
	return options.str();
}

void compile(int deviceNumber) {
	std::string options = compileOptions(deviceNumber);
	const char *optionsPtr = static_cast<const char *>(options.c_str());
	#pragma omp parallel sections
	{
	#pragma omp section
	cu::Source(SOURCE_GRIDDER).compile(PTX_GRIDDER, optionsPtr);
	#pragma omp section
	cu::Source(SOURCE_DEGRIDDER).compile(PTX_DEGRIDDER, optionsPtr);
	#pragma omp section
	cu::Source(SOURCE_ADDER).compile(PTX_ADDER, optionsPtr);
	#pragma omp section
	cu::Source(SOURCE_SHIFTER).compile(PTX_SHIFTER, optionsPtr);
	}
	#pragma omp barrier
	std::clog << std::endl;
}


/*
    Benchmark
*/
double runtime(const Record &startRecord, const Record &stopRecord) {
    return stopRecord.event.elapsedTime(startRecord.event) * 1e-3;
}

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

void report(const char *name, const Record &startRecord, const Record &stopRecord, uint64_t flops, uint64_t bytes) {
#if defined MEASURE_POWER
    double Watt = PowerSensor::Watt(startRecord.state, stopRecord.state);
#endif
    double runtime = stopRecord.event.elapsedTime(startRecord.event) * 1e-3;
	#pragma omp critical(clog)
	{
    std::clog << name << ": " << runtime << " s";
    if (flops != 0)
		std::clog << ", " << flops / runtime * 1e-12 << " TFLOPS";
    if (bytes != 0)
		std::clog << ", " << bytes / runtime * 1e-9 << " GB/s";
#if defined MEASURE_POWER
    std::clog << ", " << Watt << " W";
    if (flops != 0)
        std::clog << ", " << flops / runtime / Watt * 1e-9 << " GFLOPS/W";
#endif
    std::clog << std::endl;
	}
}

#if defined MEASURE_POWER
void report(const char *name, double runtime, uint64_t flops, PowerSensor::State startState, PowerSensor::State stopState) {
    double energy = PowerSensor::Joules(startState, stopState);

    #pragma omp critical(clog)
    std::clog << name << ": " << runtime << " s, "
              << flops / runtime * 1e-12 << " TFLOPS"
              << ", " << PowerSensor::Watt(startState, stopState) << " W"
              << ", " << flops / energy * 1e-9 << " GFLOPS/W"
              << std::endl;
}
#endif

void report_visibilities(double runtime) {
    int nr_visibilities = NR_BASELINES * NR_TIME * NR_CHANNELS;
    std::clog << "throughput: " << 1e-6 * nr_visibilities / runtime << " Mvisibilities/s" << std::endl;
}


void report_subgrids(double runtime) {
    std::clog << "throughput: " << 1e-3 * NR_BASELINES / runtime << " Ksubgrids/s" << std::endl;
}


extern "C" {

/*
	Init
*/
void init(int gpuDeviceNumber, const char *powerSensorDevice) {
    cu::init();
	printDevices(gpuDeviceNumber);
	compile(gpuDeviceNumber);
#if defined MEASURE_POWER
    std::clog << "Opening power sensor: " << powerSensorDevice << std::endl;
    powerSensor = new PowerSensor(powerSensorDevice, "powerdump");
#endif
    std::clog << std::setprecision(4);
}


/*
	Gridder
*/
void run_gridder(
	int deviceNumber, int nr_streams, int jobsize,
	void *visibilities, void *uvw, void *offset, void *wavenumbers,
	void *aterm, void *spheroidal, void *baselines, void *uvgrid) {

    // Initialize
	cu::Device device(deviceNumber);
    cu::Context context(device);
	cu::Stream globalstream;
    Record records_total[4];
	
	// Load kernel modules
	cu::Module module_gridder(PTX_GRIDDER);
	cu::Module module_shifter(PTX_SHIFTER);
	
	// Load kernel functions
	KernelGridder kernel_gridder(module_gridder, KERNEL_GRIDDER);
	KernelShifter kernel_shifter(module_shifter, KERNEL_SHIFTER);
	KernelFFT kernel_fft;
	
	// Host memory
	records_total[0].enqueue(globalstream);
	cu::HostMemory h_offset(sizeof(OffsetType), CU_MEMHOSTALLOC_WRITECOMBINED);
	cu::HostMemory h_wavenumbers(sizeof(WavenumberType), CU_MEMHOSTALLOC_WRITECOMBINED);
	cu::HostMemory h_aterm(sizeof(ATermType), CU_MEMHOSTALLOC_WRITECOMBINED);
	cu::HostMemory h_spheroidal(sizeof(SpheroidalType), CU_MEMHOSTALLOC_WRITECOMBINED);
	cu::HostMemory h_baselines(sizeof(BaselineType), CU_MEMHOSTALLOC_WRITECOMBINED);
	cu::HostMemory h_visibilities(sizeof(VisibilitiesType));
	cu::HostMemory h_uvw(sizeof(UVWType));
	cu::HostMemory h_uvgrid(sizeof(UVGridType));
	
	// Initialize host memory
	memcpy(h_offset, offset, sizeof(OffsetType));
	memcpy(h_wavenumbers, wavenumbers, sizeof(WavenumberType));
	memcpy(h_aterm, aterm, sizeof(ATermType));
	memcpy(h_spheroidal, spheroidal, sizeof(SpheroidalType));
	memcpy(h_baselines, baselines, sizeof(BaselineType));
	memcpy(h_visibilities, visibilities, sizeof(VisibilitiesType));
	memcpy(h_uvw, uvw, sizeof(UVWType));
	memset(h_uvgrid, 0, sizeof(UVGridType));
	
	// Device memory
	cu::DeviceMemory d_offset(sizeof(OffsetType));
	cu::DeviceMemory d_wavenumbers(sizeof(WavenumberType));
	cu::DeviceMemory d_aterm(sizeof(ATermType));
	cu::DeviceMemory d_spheroidal(sizeof(SpheroidalType));
	cu::DeviceMemory d_baselines(sizeof(BaselineType));

	// Copy static datastructures to device
    globalstream.memcpyHtoDAsync(d_offset, h_offset);
	globalstream.memcpyHtoDAsync(d_wavenumbers, h_wavenumbers);
	globalstream.memcpyHtoDAsync(d_aterm, h_aterm);
	globalstream.memcpyHtoDAsync(d_spheroidal, h_spheroidal);
	globalstream.memcpyHtoDAsync(d_baselines, h_baselines);
	records_total[1].enqueue(globalstream);
	globalstream.synchronize();

    // Timing variables
    double total_time_gridder[nr_streams];
    double total_time_shifter[nr_streams];
    double total_time_fft[nr_streams];
    double total_time_in[nr_streams];
    double total_time_out[nr_streams];
    double total_bytes_in[nr_streams];
    double total_bytes_out[nr_streams];
    for (int t = 0; t < nr_streams; t++) {
        total_time_gridder[t] = 0;
        total_time_shifter[t] = 0;
        total_time_fft[t] = 0;
        total_time_in[t] = 0;
        total_time_out[t] = 0;
        total_bytes_in[t] = 0;
        total_bytes_out[t] = 0;
    }
    
    // Warmup
    #if WARMUP
    cu::DeviceMemory d_uvgrid(sizeof(UVGridType)/NR_BASELINES);
    kernel_fft.launchAsync(globalstream, 1, d_uvgrid, CUFFT_FORWARD);
    kernel_fft.launchAsync(globalstream, 1, d_uvgrid, CUFFT_INVERSE);
    globalstream.synchronize();
    #endif

	// Start gridder
    records_total[2].enqueue(globalstream);
    double time_start = omp_get_wtime();
    
	#pragma omp parallel num_threads(nr_streams)
	{
	    // Initialize
	    context.setCurrent();
	    cu::Stream stream;
	    cu::Event event;
        int current_jobsize = jobsize;
	    Record records[12];
        int thread_num = omp_get_thread_num();
	    
	    // Private device memory
    	cu::DeviceMemory d_visibilities(VISIBILITY_SIZE);
    	cu::DeviceMemory d_uvw(UVW_SIZE);
	    cu::DeviceMemory d_uvgrid(UVGRID_SIZE);
	    
        #if LOOP
        while (true) {
        #endif
	    #if REPEAT
        for (int r = 0; r < NR_REPETITIONS; r++) {
        #endif
        for (int bl = thread_num * jobsize; bl < NR_BASELINES; bl += nr_streams * jobsize) {
	        // Prevent overflow
		    current_jobsize = bl + jobsize > NR_BASELINES ? NR_BASELINES - bl : jobsize;
		    
		    // Number of elements in batch
		    size_t visibility_elements = NR_TIME * NR_CHANNELS * NR_POLARIZATIONS;
		    size_t uvw_elements        = NR_TIME * 3;
		    size_t uvgrid_elements     = BLOCKSIZE * BLOCKSIZE * NR_POLARIZATIONS;
		
		    // Pointers to data for batch
		    void *visibilities_ptr = (float complex *) h_visibilities + bl * visibility_elements;
		    void *uvw_ptr          = (float *) h_uvw + bl * uvw_elements;
		    void *uvgrid_ptr       = (float complex *) h_uvgrid + bl * uvgrid_elements;
		    
	        // Copy input data to device
	        records[0].enqueue(stream);
            #if INPUT
	        stream.memcpyHtoDAsync(d_visibilities, visibilities_ptr, VISIBILITY_SIZE);
	        stream.memcpyHtoDAsync(d_uvw, uvw_ptr, UVW_SIZE);
            #endif 
            records[1].enqueue(stream);

	        // Launch gridder kernel
	        records[2].enqueue(stream);
            #if GRIDDER
            kernel_gridder.launchAsync(stream, current_jobsize, bl, d_uvw, d_offset, d_wavenumbers, d_visibilities, d_spheroidal, d_aterm, d_baselines, d_uvgrid);
            #endif
	        records[3].enqueue(stream);
	        
	        // Launch shifter kernel
	        records[4].enqueue(stream);
	        #if SHIFTER 
            kernel_shifter.launchAsync(stream, current_jobsize, d_uvgrid);
            #endif
	        records[5].enqueue(stream);

	        // Launch FFT
	        records[6].enqueue(stream);
            #if FFT
	        kernel_fft.launchAsync(stream, current_jobsize, d_uvgrid, CUFFT_INVERSE);
	        #endif
            records[7].enqueue(stream);
	        
	        // Launch shifter kernel
	        records[8].enqueue(stream);
            #if SHIFTER
	        kernel_shifter.launchAsync(stream, current_jobsize, d_uvgrid);
            #endif
	        records[9].enqueue(stream);
	        
	        // Wait for computation of uvgrid to finish
	        stream.record(event);
	        stream.waitEvent(event);
	
	        // Copy uvgrid to host
	        records[10].enqueue(stream);
            #if OUTPUT
	        stream.memcpyDtoHAsync(uvgrid_ptr, d_uvgrid, UVGRID_SIZE);
            #endif
            records[11].enqueue(stream);
            
            // Wait for memory transfers to finish
            stream.synchronize();
		
		    #if REPORT_VERBOSE
            #if INPUT
            report("  input", records[0], records[1],
			    0, UVW_SIZE + VISIBILITY_SIZE);
            #endif
            #if GRIDDER
    	    report("gridder", records[2], records[3],
  		        kernel_gridder.flops(current_jobsize), kernel_gridder.bytes(current_jobsize));
            #endif
            #if SHIFTER
            report("shifter", records[4], records[5],
			    kernel_fft.flops(current_jobsize), kernel_shifter.bytes(current_jobsize));
		    #endif
            #if FFT
            report("    fft", records[6], records[7],
			    kernel_fft.flops(current_jobsize), kernel_fft.bytes(current_jobsize));
		    #endif
            #if SHIFTER
            report("shifter", records[8], records[9],
			    kernel_fft.flops(current_jobsize), kernel_shifter.bytes(current_jobsize));
            #endif
		    #if OUTPUT 
            report(" output", records[10], records[11],
			    0, UVGRID_SIZE);
		    #endif
		    #endif
		    
		    // Update total runtime
		    total_time_in[thread_num]   += runtime(records[0], records[1]);
	        total_time_gridder[thread_num] += runtime(records[2], records[3]);
	        total_time_shifter[thread_num] += runtime(records[4], records[5]);
	        total_time_fft[thread_num]     += runtime(records[6], records[7]);
	        total_time_shifter[thread_num] += runtime(records[8], records[9]);
	        total_time_out[thread_num]  += runtime(records[10], records[11]);
	        total_bytes_in[thread_num]        += UVW_SIZE + VISIBILITY_SIZE;
	        total_bytes_out[thread_num]       += UVGRID_SIZE;
	    }
	    #if LOOP
        }
        #endif
	    #if REPEAT
        }
        #endif
	}

    // Measure total runtime
    records_total[3].enqueue(globalstream);
    globalstream.synchronize();

    // Copy uvgrids
    memcpy(uvgrid, h_uvgrid, sizeof(UVGridType));
    
    #if REPORT_TOTAL
    std::clog << std::endl;
    // Report performance per stream
    for (int t = 0; t < nr_streams; t++) {
        std::clog << "--- stream " << t << " ---" << std::endl;
        int jobsize = NR_BASELINES / nr_streams;
        report("gridder", total_time_gridder[t],
                          kernel_gridder.flops(jobsize),
                          kernel_gridder.bytes(jobsize));
	    report("shifter", total_time_shifter[t]/2,
	                      kernel_shifter.flops(jobsize),
	                      kernel_shifter.bytes(jobsize));
        report("    fft", total_time_fft[t], kernel_fft.flops(jobsize), kernel_fft.bytes(jobsize));
        report("  input", total_time_in[t], 0, total_bytes_in[t]);
        report(" output", total_time_out[t], 0, total_bytes_out[t]);
	    std::clog << std::endl;
    }
    
    // Report overall performance
    std::clog << "--- overall ---" << std::endl;
    long total_flops = kernel_gridder.flops(NR_BASELINES) + kernel_shifter.flops(NR_BASELINES)*2 + kernel_fft.flops(NR_BASELINES);
    report("      init", records_total[0], records_total[1], 0, 0);
    report("     total", records_total[2], records_total[3], total_flops, 0);
    double total_runtime = runtime(records_total[2], records_total[3]);
    report_visibilities(total_runtime);
    std::clog << std::endl;
    #endif
}


/*
	Adder
*/
void run_adder(int deviceNumber, int nr_streams, int jobsize,
	void *coordinates, void *uvgrid, void *grid) {
	// Initialize
	cu::Device device(deviceNumber);
    cu::Context context(device);
	
	// Load kernel module
	cu::Module module_adder(PTX_ADDER);
	
	// Load kernel function
	KernelAdder kernel_adder(module_adder, KERNEL_ADDER);

	// Streams
	cu::Stream globalstream;
	
	// Host memory
	cu::HostMemory h_coordinates(sizeof(CoordinateType), CU_MEMHOSTALLOC_WRITECOMBINED);
	cu::HostMemory h_uvgrid(sizeof(UVGridType), CU_MEMHOSTALLOC_WRITECOMBINED);
	cu::HostMemory h_grid(sizeof(GridType));
	
	// Initialize host memory
	memcpy(h_coordinates, coordinates, sizeof(CoordinateType));
	memcpy(h_uvgrid, uvgrid, sizeof(UVGridType));
	
	// Device memory
	cu::DeviceMemory d_coordinates(sizeof(CoordinateType));
	cu::DeviceMemory d_grid(sizeof(GridType));
	d_grid.zero();
	
	// Copy static datastructures to device
	globalstream.memcpyHtoDAsync(d_coordinates, h_coordinates);
	globalstream.synchronize();
	
	// Timing variables
	double total_time_adder[nr_streams];
	double total_time_in[nr_streams];
	double total_time_out = 0;
	double total_bytes_in[nr_streams];
	double total_bytes_out[nr_streams];
    for (int t = 0; t < nr_streams; t++) {
        total_time_adder[t] = 0;
        total_time_in[t] = 0;
        total_bytes_in[t] = 0;
        total_bytes_out[t] = 0;
    }
	
	// Start adder
    Record records_total[2];
    records_total[0].enqueue(globalstream);
	#pragma omp parallel num_threads(nr_streams)
	{
	    // Initialize
	    context.setCurrent();
        cu::Stream stream;
        cu::Event event;
        Record records[4];
	    int current_jobsize = jobsize;
	    int thread_num = omp_get_thread_num();
        
        // Private device memory
    	cu::DeviceMemory d_uvgrid(UVGRID_SIZE);
    
	    for (int bl = thread_num * jobsize; bl < NR_BASELINES; bl += nr_streams * jobsize) {
	        // Prevent overflow
		    current_jobsize = bl + jobsize > NR_BASELINES ? NR_BASELINES - bl : jobsize;
           
		    // Number of elements in batch
		    size_t uvgrid_elements = BLOCKSIZE * BLOCKSIZE * NR_POLARIZATIONS;
		
		    // Pointers to data for current batch
		    void *uvgrid_ptr = (float complex *) h_uvgrid + bl * uvgrid_elements;

	        // Copy input to device
            records[0].enqueue(stream);
            #if INPUT
	        stream.memcpyHtoDAsync(d_uvgrid, uvgrid_ptr, UVGRID_SIZE);
	        #endif
            records[1].enqueue(stream);

	        // Launch add kernel
            records[2].enqueue(stream);
            #if ADDER
	        kernel_adder.launchAsync(stream, current_jobsize, bl, d_coordinates, d_uvgrid, d_grid);
	        #endif
            records[3].enqueue(stream);

	        // Wiat for computation of grid to finish
	        stream.record(event);
	        event.synchronize();
            
		    #if REPORT_VERBOSE
		    #if INPUT
		    report("  input", records[0], records[1], 0, UVGRID_SIZE);
		    #endif
		    #if ADDER
		    report("  adder", records[2], records[3], kernel_adder.flops(current_jobsize), kernel_adder.bytes(current_jobsize));
		    #endif
		    #endif
		    
		    // Update total runtime
		    total_time_in[thread_num] += runtime(records[0], records[1]);
		    total_time_adder[thread_num] += runtime(records[2], records[3]);
		    total_bytes_in[thread_num]      += UVGRID_SIZE;
	    }
	}
	
    // Copy grid to host
    Record records[2];
    records[0].enqueue(globalstream);
    globalstream.memcpyDtoHAsync(h_grid, d_grid, sizeof(GridType));
    records[1].enqueue(globalstream);
    globalstream.synchronize();
    memcpy(grid, h_grid, sizeof(GridType));
    total_time_out = runtime(records[0], records[1]);

    // Measure total runtime
    records_total[1].enqueue(globalstream);
    globalstream.synchronize();

    #if REPORT_TOTAL
    std::clog << std::endl;
    
    // Report performance per stream
    for (int t = 0; t < nr_streams; t++) {
        std::clog << "--- stream " << t << " ---" << std::endl;
        int jobsize = NR_BASELINES / nr_streams;
        report("  adder", total_time_adder[t],
                          kernel_adder.flops(jobsize),
                          kernel_adder.bytes(jobsize));
        report("  input", total_time_in[t], 0, sizeof(UVGridType));
        std::clog << std::endl;
    }
    
    // Report overall performance
    std::clog << "--- overall ---" << std::endl;
    long total_flops = kernel_adder.flops(NR_BASELINES);
    report(" output", total_time_out, 0, sizeof(GridType));
    report("  total", records_total[0], records_total[1], total_flops, 0);
    double total_runtime = runtime(records_total[0], records_total[1]);
    report_subgrids(total_runtime);
    std::clog << std::endl;
    #endif
}


/*
	Degridder
*/
void run_degridder(
	int deviceNumber, int nr_streams, int jobsize,
	void *offset, void *wavenumbers, void *aterm, void *baselines,
	void *visibilities, void *uvw, void *spheroidal, void *uvgrid) {

    // Initialize
	cu::Device device(deviceNumber);
    cu::Context context(device);
    cu::Stream globalstream;
    Record records_total[4];
	
	// Load kernel modules
	cu::Module module_degridder(PTX_DEGRIDDER);
	cu::Module module_shifter(PTX_SHIFTER);
	
	// Load kernel functions
	KernelDegridder kernel_degridder(module_degridder, KERNEL_DEGRIDDER);
	KernelFFT kernel_fft;
	KernelShifter kernel_shifter(module_shifter, KERNEL_SHIFTER);
	
	// Host memory
    records_total[0].enqueue(globalstream);
	cu::HostMemory h_offset(sizeof(OffsetType), CU_MEMHOSTALLOC_WRITECOMBINED);
	cu::HostMemory h_wavenumbers(sizeof(WavenumberType), CU_MEMHOSTALLOC_WRITECOMBINED);
	cu::HostMemory h_aterm(sizeof(ATermType), CU_MEMHOSTALLOC_WRITECOMBINED);
	cu::HostMemory h_baselines(sizeof(BaselineType), CU_MEMHOSTALLOC_WRITECOMBINED);
	cu::HostMemory h_uvw(sizeof(UVWType), CU_MEMHOSTALLOC_WRITECOMBINED);
	cu::HostMemory h_visibilities(sizeof(VisibilitiesType));
	cu::HostMemory h_spheroidal(sizeof(SpheroidalType));
	cu::HostMemory h_uvgrid(sizeof(UVGridType), CU_MEMHOSTALLOC_WRITECOMBINED);
	
	// Initialize host memory
	memcpy(h_offset, offset, sizeof(OffsetType));
	memcpy(h_wavenumbers, wavenumbers, sizeof(WavenumberType));
	memcpy(h_aterm, aterm, sizeof(ATermType));
	memcpy(h_baselines, baselines, sizeof(BaselineType));
	memcpy(h_uvw, uvw, sizeof(UVWType));
	memset(h_visibilities, 0, sizeof(VisibilitiesType));
	memcpy(h_spheroidal, spheroidal, sizeof(SpheroidalType));
	memcpy(h_uvgrid, uvgrid, sizeof(UVGridType));
	
	// Device memory
	cu::DeviceMemory d_offset(sizeof(OffsetType));
	cu::DeviceMemory d_wavenumbers(sizeof(WavenumberType));
	cu::DeviceMemory d_aterm(sizeof(ATermType));
	cu::DeviceMemory d_baselines(sizeof(BaselineType));
	cu::DeviceMemory d_spheroidal(sizeof(SpheroidalType));
	
	// Copy static datastructures to device
	globalstream.memcpyHtoDAsync(d_offset, h_offset);
	globalstream.memcpyHtoDAsync(d_wavenumbers, h_wavenumbers);
	globalstream.memcpyHtoDAsync(d_aterm, h_aterm);
	globalstream.memcpyHtoDAsync(d_baselines, h_baselines);
	globalstream.memcpyHtoDAsync(d_spheroidal, h_spheroidal);
    records_total[1].enqueue(globalstream);
	globalstream.synchronize();

    // Timing variables
	double total_time_degridder[nr_streams];
	double total_time_shifter[nr_streams];
	double total_time_fft[nr_streams];
	double total_time_in[nr_streams];
	double total_time_out[nr_streams];
    double total_bytes_in[nr_streams];
    double total_bytes_out[nr_streams];
    for (int t = 0; t < nr_streams; t++) {
        total_time_degridder[t] = 0;
        total_time_shifter[t] = 0;
        total_time_fft[t] = 0;
        total_time_in[t] = 0;
        total_time_out[t] = 0;
        total_bytes_in[t] = 0;
        total_bytes_out[t] = 0;
    }

    // Warmup
    #if WARMUP
    cu::DeviceMemory d_uvgrid(sizeof(UVGridType)/NR_BASELINES);
    kernel_fft.launchAsync(globalstream, 1, d_uvgrid, CUFFT_FORWARD);
    kernel_fft.launchAsync(globalstream, 1, d_uvgrid, CUFFT_INVERSE);
    globalstream.synchronize();
    #endif

    // Start degridder
    records_total[2].enqueue(globalstream);
	double time_start = omp_get_wtime();
    
    #pragma omp parallel num_threads(nr_streams)
	{
	    // Initialize
	    context.setCurrent();
        cu::Stream stream;
        cu::Event event;
	    int current_jobsize = jobsize;
        Record records[12];
        int thread_num = omp_get_thread_num();
    
	    // Private memory
    	cu::DeviceMemory d_uvw(UVW_SIZE);
    	cu::DeviceMemory d_visibilities(VISIBILITY_SIZE);
    	cu::DeviceMemory d_uvgrid(UVGRID_SIZE);

        #if LOOP 
        while (true) {
        #endif
        #if REPEAT
        for (int r = 0; r < NR_REPETITIONS; r++) {
        #endif
	    for (int bl = thread_num * jobsize; bl < NR_BASELINES; bl += nr_streams * jobsize) {
	        // Prevent overflow
		    current_jobsize = bl + jobsize > NR_BASELINES ? NR_BASELINES - bl : jobsize;

		    // Number of elements in batch
		    size_t uvw_elements        = NR_TIME * 3;
		    size_t visibility_elements = NR_TIME * NR_CHANNELS * NR_POLARIZATIONS;
		    size_t uvgrid_elements     = BLOCKSIZE * BLOCKSIZE * NR_POLARIZATIONS;
		
		    // Pointers to data for current batch
		    void *visibilities_ptr = (float complex *) h_visibilities + bl * visibility_elements;
		    void *uvw_ptr          = (float *) h_uvw + bl * uvw_elements;
		    void *uvgrid_ptr       = (float complex *) h_uvgrid + bl * uvgrid_elements;
		
	        // Copy input data to device
	        records[0].enqueue(stream);
            #if INPUT
	        stream.memcpyHtoDAsync(d_uvw, uvw_ptr, UVW_SIZE);
	        stream.memcpyHtoDAsync(d_uvgrid, uvgrid_ptr, UVGRID_SIZE);
            #endif
	        records[1].enqueue(stream);

	        // Run shifter kernel
	        records[2].enqueue(stream);
	        #if SHIFTER
            kernel_shifter.launchAsync(stream, current_jobsize, d_uvgrid);
            #endif
	        records[3].enqueue(stream);
	
	        // Run FFT
	        records[4].enqueue(stream);
            #if FFT
	        kernel_fft.launchAsync(stream, current_jobsize, d_uvgrid, CUFFT_FORWARD);
            #endif
	        records[5].enqueue(stream);
	        
	        // Run shifter kernel
	        records[6].enqueue(stream);
            #if SHIFTER
	        kernel_shifter.launchAsync(stream, current_jobsize, d_uvgrid);
            #endif
	        records[7].enqueue(stream);
	
	        // Launch degridder kernel
	        records[8].enqueue(stream);
            #if DEGRIDDER
	        kernel_degridder.launchAsync(stream, current_jobsize, bl,
		        d_uvgrid, d_uvw, d_offset, d_wavenumbers, d_aterm,
                d_baselines, d_spheroidal, d_visibilities);
            #endif
	        records[9].enqueue(stream);

	        // Copy visibilities to host
	        records[10].enqueue(stream);
            #if OUTPUT
	        stream.memcpyDtoHAsync(visibilities_ptr, d_visibilities, VISIBILITY_SIZE);
            #endif
	        records[11].enqueue(stream);
			
	        // Wait for memory transfers to finish
            stream.record(event);
	        event.synchronize();
	
            #if REPORT_VERBOSE
	        #if INPUT
            report("    input", records[0], records[1], 0, UVW_SIZE + UVGRID_SIZE);
	        #endif
            #if SHIFTER
            report("  shifter", records[2], records[3],
	            kernel_shifter.flops(current_jobsize), kernel_shifter.bytes(current_jobsize));
	        #endif
            #if FFT
            report("      fft", records[4], records[5], 
                kernel_fft.flops(current_jobsize), kernel_fft.bytes(current_jobsize));
	        #endif
            #if SHIFTER
            report("  shifter", records[6], records[7],
	            kernel_shifter.flops(current_jobsize), kernel_shifter.bytes(current_jobsize));
	        #endif
            #if DEGRIDDER
            report("degridder", records[8], records[9],
		        kernel_degridder.flops(current_jobsize), kernel_degridder.bytes(current_jobsize));
            #endif
            #if OUTPUT
	        report("   output", records[10], records[11], 0, VISIBILITY_SIZE);
            #endif
	        #endif
	        
             // Update total runtime
             total_time_in[thread_num]     += runtime(records[0], records[1]);
             total_time_shifter[thread_num]   += runtime(records[2], records[3]);
             total_time_fft[thread_num]       += runtime(records[4], records[5]);
             total_time_shifter[thread_num]   += runtime(records[6], records[7]);
             total_time_degridder[thread_num] += runtime(records[8], records[9]);
             total_time_out[thread_num]    += runtime(records[10], records[11]);
             total_bytes_in[thread_num]          += UVW_SIZE + UVGRID_SIZE;
             total_bytes_out[thread_num]         += VISIBILITY_SIZE;
	    }
	    #if LOOP
        }
        #endif
        #if REPEAT
        }
        #endif
    }
    
    // Copy visibilities
    memcpy(visibilities, h_visibilities, sizeof(VisibilitiesType));

    // Measure total runtime
    records_total[3].enqueue(globalstream);
    globalstream.synchronize();

    #if REPORT_TOTAL

    // Pinrt report per thread
    jobsize = NR_BASELINES / nr_streams;
    std::clog << std::endl;
    for (int t = 0; t < nr_streams; t++) {
        std::clog << "--- stream " << t << " ---" << std::endl;
        report("      fft", total_time_fft[t],
                            kernel_fft.flops(jobsize),
                            kernel_fft.bytes(jobsize));
        report("  shifter", total_time_shifter[t]/2,
                            kernel_shifter.flops(jobsize),
                            kernel_shifter.bytes(jobsize));
        report("degridder", total_time_degridder[t],
                            kernel_degridder.flops(jobsize),
                            kernel_degridder.bytes(jobsize));
        report("    input", total_time_in[t], 0,
                            total_bytes_in[t]);
        report("   output", total_time_out[t], 0,
                            total_bytes_out[t]);
        std::clog << std::endl;
    }
    
    // Print overall report
    std::clog << "--- overall ---" << std::endl;
    long total_flops = kernel_shifter.flops(NR_BASELINES)*2 +
                       kernel_degridder.flops(NR_BASELINES) +
                       kernel_fft.flops(NR_BASELINES);
    report("     init", records_total[0], records_total[1], 0, 0);
    report("    total", records_total[2], records_total[3], total_flops, 0);
    double total_runtime = runtime(records_total[2], records_total[3]);
    report_visibilities(total_runtime);
    std::clog << std::endl;
    #endif
}
}
