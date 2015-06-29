#include <omp.h>
#include <complex.h>

#include <sstream>

#include "Types.h"
#include "Kernels.h"
#include "Power.h"


/*
    Performance reporting
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
#define SHIFTER   1
#define FFT       1
#define INPUT     1
#define OUTPUT    1


/*
	File and kernel names
*/
#define SOURCE_GRIDDER      "Gridder.cu"
#define SOURCE_DEGRIDDER    "Degridder.cu"
#define SOURCE_ADDER        "Adder.cu"
#define SOURCE_SPLITTER     "Splitter.cu"
#define PTX_DEGRIDDER       "Degridder.ptx"
#define PTX_GRIDDER         "Gridder.ptx"
#define PTX_ADDER 	    "Adder.ptx"
#define PTX_SPLITTER        "Splitter.ptx"
#define KERNEL_DEGRIDDER    "kernel_degridder"
#define KERNEL_GRIDDER      "kernel_gridder"
#define KERNEL_ADDER        "kernel_adder"
#define KERNEL_SPLITTER     "kernel_splitter"


/*
    Size of device datastructures for one block of work
*/
#define VISIBILITY_SIZE	current_jobsize * NR_TIME * NR_CHANNELS * NR_POLARIZATIONS * sizeof(float2)
#define UVW_SIZE		current_jobsize * NR_TIME * 3 * sizeof(float)
#define SUBGRID_SIZE	current_jobsize * NR_CHUNKS * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * sizeof(float2)


/*
    Benchmark
*/
struct Record
{
  public:
    void enqueue(cu::Stream&);
    mutable cu::Event event;
};

void Record::enqueue(cu::Stream &stream) {
    stream.record(event);
}

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
    double runtime = stopRecord.event.elapsedTime(startRecord.event) * 1e-3;
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


extern "C" {
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
	options << " -DSUBGRIDSIZE="		<< SUBGRIDSIZE;
	options << " -DGRIDSIZE="			<< GRIDSIZE;
	options << " -DCHUNKSIZE="          << CHUNKSIZE;
	options << " -DIMAGESIZE="			<< IMAGESIZE;
	int capability = 10 * device.getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR>() +
						  device.getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR>();
	options << " -arch=compute_" << capability;
	options << " -code=sm_" << capability;
	options << " -use_fast_math";
    options << " -lineinfo";
    options << " -src-in-ptx";
    options << " -I../Common";
    options << " -g";
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
	cu::Source(SOURCE_SPLITTER).compile(PTX_SPLITTER, optionsPtr);
	}
	#pragma omp barrier
	std::clog << std::endl;
}


/*
    State
*/
class State {
    public:
        State();
        int jobsize;
        Record startRecord;
        Record stopRecord;
        double runtime;
        double flops;
        double bytes;
        double power;
};

State::State() {
    jobsize = 0;
    runtime = 0;
    flops   = 0;
    bytes   = 0;
    power   = 0;
}


/*
	Gridder
*/
void callback_gridder_input(CUstream, CUresult, void *userData) {
    State *s = (State *) userData;
    int current_jobsize = s->jobsize;
    s->runtime = runtime(s->startRecord, s->stopRecord);
    s->flops = 0;
    s->bytes = UVW_SIZE + VISIBILITY_SIZE;
    #if MEASURE_POWER
    s->power = power(s->startRecord, s->stopRecord);
    #endif
    #if REPORT_VERBOSE
    report("  input", s->startRecord, s->stopRecord, s->flops, s->bytes);
    #endif
}

void callback_gridder_gridder(CUstream, CUresult, void *userData) {
     State *s = (State *) userData;
    int current_jobsize = s->jobsize;
    s->runtime = runtime(s->startRecord, s->stopRecord);
    s->flops = KernelGridder::flops(current_jobsize);
    s->bytes = KernelGridder::bytes(current_jobsize);
    #if MEASURE_POWER
    s->power = power(s->startRecord, s->stopRecord);
    #endif
    #if REPORT_VERBOSE
    report("gridder", s->startRecord, s->stopRecord, s->flops, s->bytes);
    #endif
}

void callback_gridder_fft(CUstream, CUresult, void *userData) {
     State *s = (State *) userData;
    int current_jobsize = s->jobsize;
    s->runtime = runtime(s->startRecord, s->stopRecord);
    s->flops = KernelFFT::flops(SUBGRIDSIZE, current_jobsize);
    s->bytes = KernelFFT::bytes(SUBGRIDSIZE, current_jobsize);
    #if MEASURE_POWER
    s->power = power(s->startRecord, s->stopRecord);
    #endif
    #if REPORT_VERBOSE
    report("    fft", s->startRecord, s->stopRecord, s->flops, s->bytes);
    #endif
}

void callback_gridder_output(CUstream, CUresult, void *userData) {
    State *s = (State *) userData;
    int current_jobsize = s->jobsize;
    s->runtime = runtime(s->startRecord, s->stopRecord);
    s->flops = 0;
    s->bytes = SUBGRID_SIZE;
    #if MEASURE_POWER
    s->power = power(s->startRecord, s->stopRecord);
    #endif
    #if REPORT_VERBOSE
    report(" output", s->startRecord, s->stopRecord, s->flops, s->bytes);
    #endif
}

void run_gridder(
    cu::Context &context, int nr_streams, int jobsize,
    cu::HostMemory &h_visibilities, cu::HostMemory &h_uvw,
    cu::HostMemory &h_subgrid, cu::DeviceMemory &d_wavenumbers,
    cu::DeviceMemory &d_spheroidal, cu::DeviceMemory &d_aterm,
    cu::DeviceMemory &d_baselines) {

    // Initialize
    Record records_total[2];
    cu::Stream executestream;
    cu::Stream htodstream;
    cu::Stream dtohstream;
	
	// Load kernel module
	cu::Module module_gridder(PTX_GRIDDER);
	
    // Load kernel function
	KernelGridder kernel_gridder(module_gridder, KERNEL_GRIDDER);
	
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
    records_total[0].enqueue(dtohstream);
	#pragma omp parallel num_threads(nr_streams)
	{
	    // Initialize
	    context.setCurrent();
        cu::Event executeFinished;
	    cu::Event inputFree;
        int current_jobsize = jobsize;
        int thread_num = omp_get_thread_num();
        KernelFFT kernel_fft;
        int iteration = 0;
	
        // Initialize states
        State states_input[nr_iterations];
        State states_gridder[nr_iterations];
        State states_fft[nr_iterations];
        State states_output[nr_iterations];
	    
	    // Private device memory
    	cu::DeviceMemory d_visibilities(VISIBILITY_SIZE);
    	cu::DeviceMemory d_uvw(UVW_SIZE);
	    cu::DeviceMemory d_subgrid(SUBGRID_SIZE);
	    
        for (int bl = thread_num * jobsize; bl < NR_BASELINES; bl += nr_streams * jobsize) {
            // Prevent overflow
		    current_jobsize = bl + jobsize > NR_BASELINES ? NR_BASELINES - bl : jobsize;
        
            // States
            State &state_input    = states_input[iteration];
            State &state_gridder  = states_gridder[iteration];
            State &state_fft      = states_fft[iteration];
            State &state_output   = states_output[iteration];
           
            // Set jobsize in states 
            state_input.jobsize    = current_jobsize;
            state_gridder.jobsize  = current_jobsize;
            state_fft.jobsize      = current_jobsize;
            state_output.jobsize   = current_jobsize;

		    // Number of elements in batch
		    size_t visibility_elements = NR_TIME * NR_CHANNELS * NR_POLARIZATIONS;
		    size_t uvw_elements        = NR_TIME * 3;
		    size_t subgrid_elements    = SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS;
		
		    // Pointers to data for batch
		    void *visibilities_ptr = (float complex *) h_visibilities + bl * visibility_elements;
		    void *uvw_ptr          = (float *) h_uvw + bl * uvw_elements;
		    void *subgrid_ptr      = (float complex *) h_subgrid + bl * subgrid_elements;

            // Wait for previous computation to finish
            executestream.record(executeFinished);
            htodstream.waitEvent(executeFinished);
		    
	        // Copy input data to device
            #pragma omp critical
            {
                inputFree.synchronize();
                htodstream.waitEvent(inputFree);
                state_input.startRecord.enqueue(htodstream);
                #if INPUT
                htodstream.memcpyHtoDAsync(d_visibilities, visibilities_ptr, VISIBILITY_SIZE);
                htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, UVW_SIZE);
                #endif 
                state_input.stopRecord.enqueue(htodstream);
                htodstream.addCallback(&callback_gridder_input, &state_input);
            }

            // Create FFT plan
            #if FFT
            #if ORDER == ORDER_BL_V_U_P
            kernel_fft.plan(SUBGRIDSIZE, current_jobsize, FFT_LAYOUT_YXP);
            #elif ORDER == ORDER_BL_P_V_U
            kernel_fft.plan(SUBGRIDSIZE, current_jobsize, FFT_LAYOUT_PYX);
            #endif
            #endif

	        #pragma omp critical
            {
                // Launch gridder kernel
                executestream.waitEvent(state_input.stopRecord.event); 
                state_gridder.startRecord.enqueue(executestream);
                #if GRIDDER
                kernel_gridder.launchAsync(
                    executestream, current_jobsize, bl, d_uvw, d_wavenumbers,
                    d_visibilities, d_spheroidal, d_aterm, d_baselines, d_subgrid);
                #endif
	            state_gridder.stopRecord.enqueue(executestream);
	            executestream.record(inputFree);
                executestream.addCallback(&callback_gridder_gridder, &state_gridder);
                 
                // Launch FFT
                state_fft.startRecord.enqueue(executestream);
                #if FFT
                kernel_fft.launchAsync(executestream, d_subgrid, CUFFT_INVERSE);
                #endif
                state_fft.stopRecord.enqueue(executestream);
                executestream.addCallback(&callback_gridder_fft, &state_fft);
            }
	        
	        #pragma omp critical 
            {
                // Copy subgrid to host
	            dtohstream.waitEvent(state_fft.stopRecord.event);
                state_output.startRecord.enqueue(dtohstream);
                #if OUTPUT
	            dtohstream.memcpyDtoHAsync(subgrid_ptr, d_subgrid, SUBGRID_SIZE);
                #endif
                state_output.stopRecord.enqueue(dtohstream);
                dtohstream.addCallback(&callback_gridder_output, &state_output);
            }

            // Go to next iteration
            total_jobs[thread_num] += current_jobsize;
            iteration++;
        }

        // Sum totals
        #if REPORT_TOTAL
        for (int i = 0; i < nr_iterations; i++) {
            total_time_gridder[thread_num] += states_gridder[i].runtime;
            total_time_fft[thread_num]     += states_fft[i].runtime;
            total_time_input[thread_num]   += states_input[i].runtime;
            total_time_output[thread_num]  += states_output[i].runtime;
            total_bytes_input[thread_num]  += states_input[i].bytes;
            total_bytes_output[thread_num] += states_output[i].bytes;
        }
        #endif
        
        // Wait for final transfer to finish 
        dtohstream.synchronize();
	}

    // Measure total runtime
    records_total[1].enqueue(dtohstream);
    dtohstream.synchronize();

    #if REPORT_TOTAL
    std::clog << std::endl;
    KernelFFT kernel_fft;
    // Report performance per stream
    for (int t = 0; t < nr_streams; t++) {
        std::clog << "--- stream " << t << " ---" << std::endl;
        int jobsize = total_jobs[t];
        report("gridder", total_time_gridder[t],
                          kernel_gridder.flops(jobsize),
                          kernel_gridder.bytes(jobsize));
        report("    fft", total_time_fft[t],
                          kernel_fft.flops(SUBGRIDSIZE, jobsize),
                          kernel_fft.bytes(SUBGRIDSIZE, jobsize));
        report("  input", total_time_input[t], 0, total_bytes_input[t]);
        report(" output", total_time_output[t], 0, total_bytes_output[t]);
	    std::clog << std::endl;
    }
    
    // Report overall performance
    std::clog << "--- overall ---" << std::endl;
    long total_flops = kernel_gridder.flops(NR_BASELINES) +
                       kernel_fft.flops(SUBGRIDSIZE, NR_BASELINES);
    report("     total", records_total[0], records_total[1], total_flops, 0);
    double total_runtime = runtime(records_total[0], records_total[1]);
    report_visibilities(total_runtime);
    std::clog << std::endl;
    #endif
}


/*
	Adder
*/
void run_adder(
    cu::Context &context, int nr_streams, int jobsize,
    cu::HostMemory &h_subgrid, cu::HostMemory &h_uvw,
    cu::HostMemory &h_grid) {
	// Load kernel module
	cu::Module module_adder(PTX_ADDER);
	
	// Load kernel function
	KernelAdder kernel_adder(module_adder, KERNEL_ADDER);

	// Streams
	cu::Stream executestream;
    cu::Stream iostream;
	
	// Allocate device memory for grid
	cu::DeviceMemory d_grid(sizeof(GridType));
	d_grid.zero();
	
	// Timing variables
	double total_time_adder[nr_streams];
	double total_time_in[nr_streams];
	double total_time_out = 0;
	double total_bytes_in[nr_streams];
	double total_bytes_out[nr_streams];
	long total_jobs[nr_streams];
    for (int t = 0; t < nr_streams; t++) {
        total_time_adder[t] = 0;
        total_time_in[t]    = 0;
        total_bytes_in[t]   = 0;
        total_bytes_out[t]  = 0;
        total_jobs[t]       = 0;
    }
	
	// Start adder
    Record records_total[2];
    records_total[0].enqueue(iostream);
	#pragma omp parallel num_threads(nr_streams)
	{
	    // Initialize
	    context.setCurrent();
        cu::Event event;
        Record records[4];
	    int current_jobsize = jobsize;
	    int thread_num = omp_get_thread_num();
        
        // Private device memory
        cu::DeviceMemory d_uvw(UVW_SIZE);
    	cu::DeviceMemory d_subgrid(SUBGRID_SIZE);
    
	    for (int bl = thread_num * jobsize; bl < NR_BASELINES; bl += nr_streams * jobsize) {
	        // Prevent overflow
		    current_jobsize = bl + jobsize > NR_BASELINES ? NR_BASELINES - bl : jobsize;
           
		    // Number of elements in batch
		    size_t uvw_elements     = NR_TIME * 3;
		    size_t subgrid_elements = SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS;
		
		    // Pointers to data for current batch
		    void *uvw_ptr     = (float *) h_uvw + bl * uvw_elements;
		    void *subgrid_ptr = (float complex *) h_subgrid + bl * subgrid_elements;

	        // Copy input to device
            records[0].enqueue(iostream);
            #if INPUT
            iostream.memcpyHtoDAsync(d_uvw, uvw_ptr, UVW_SIZE);
	        iostream.memcpyHtoDAsync(d_subgrid, subgrid_ptr, SUBGRID_SIZE);
	        #endif
            records[1].enqueue(iostream);

            // Wait for memory transfer to finish
            iostream.record(event);
            executestream.waitEvent(event);

	        // Launch add kernel
            records[2].enqueue(executestream);
            #if ADDER
	        kernel_adder.launchAsync(
	            executestream, current_jobsize, bl, d_uvw, d_subgrid, d_grid);
	        #endif
            records[3].enqueue(executestream);

	        // Wiat for computation of grid to finish
	        executestream.record(event);
	        event.synchronize();
            
		    #if REPORT_VERBOSE
		    #if INPUT
		    report("  input", records[0], records[1], 0, UVW_SIZE + SUBGRID_SIZE);
		    #endif
		    #if ADDER
		    report("  adder", records[2], records[3],
		        kernel_adder.flops(current_jobsize),
		        kernel_adder.bytes(current_jobsize));
		    #endif
		    #endif
		    
		    // Update total runtime
		    total_time_in[thread_num]    += runtime(records[0], records[1]);
		    total_time_adder[thread_num] += runtime(records[2], records[3]);
		    total_bytes_in[thread_num]   += UVW_SIZE + SUBGRID_SIZE;
		    total_jobs[thread_num]       += jobsize;
	    }
	}

    // Wait for grid computation to finish
    cu::Event event;
    executestream.record(event);
    iostream.waitEvent(event);
	
    // Copy grid to host
    Record records_output[2];
    records_output[0].enqueue(iostream);
    iostream.memcpyDtoHAsync(h_grid, d_grid, sizeof(GridType));
    records_output[1].enqueue(iostream);

    // Measure total runtime
    records_total[1].enqueue(iostream);
    iostream.synchronize();

    #if REPORT_TOTAL
    std::clog << std::endl;
    
    // Report performance per stream
    for (int t = 0; t < nr_streams; t++) {
        std::clog << "--- stream " << t << " ---" << std::endl;
        int jobsize = total_jobs[t];
        report("  input", total_time_in[t], 0, total_bytes_in[t]);
        report("  adder", total_time_adder[t],
                          kernel_adder.flops(jobsize),
                          kernel_adder.bytes(jobsize));
        std::clog << std::endl;
    }
    
    // Report overall performance
    std::clog << "--- overall ---" << std::endl;
    report(" output", records_output[0], records_output[1], 0, sizeof(GridType));
    report("  total", records_total[0], records_total[1], 0, 0);
    double total_runtime = runtime(records_total[0], records_total[1]);
    report_subgrids(total_runtime);
    std::clog << std::endl;
    #endif
}


/*
	Splitter
*/
void run_splitter(
    cu::Context &context, int nr_streams, int jobsize,
    cu::HostMemory &h_subgrid, cu::HostMemory &h_uvw,
    cu::HostMemory &h_grid) {
	// Load kernel module
	cu::Module module_splitter(PTX_SPLITTER);
	
	// Load kernel function
	KernelSplitter kernel_splitter(module_splitter, KERNEL_SPLITTER);

	// Streams
	cu::Stream executestream;
    cu::Stream htodstream;
    cu::Stream dtohstream;
	
	// Timing variables
	double total_time_splitter[nr_streams];
	double total_time_in[nr_streams];
	double total_time_out[nr_streams];
	double total_bytes_in[nr_streams];
	double total_bytes_out[nr_streams];
	long total_jobs[nr_streams];
    for (int t = 0; t < nr_streams; t++) {
        total_time_in[t]       = 0;
        total_time_splitter[t] = 0;
        total_time_out[t]      = 0;
        total_bytes_in[t]      = 0;
        total_bytes_out[t]     = 0;
        total_jobs[t]          = 0;
    }
    
    // Copy grid to device
    Record records_input[2];
	cu::DeviceMemory d_grid(sizeof(GridType));
	records_input[0].enqueue(htodstream);
	htodstream.memcpyHtoDAsync(d_grid, h_grid);
	records_input[1].enqueue(htodstream);
	executestream.synchronize();
	
	// Start adder
    Record records_total[2];
    records_total[0].enqueue(htodstream);
	#pragma omp parallel num_threads(nr_streams)
	{
	    // Initialize
	    context.setCurrent();
        cu::Event event;
        Record records[6];
	    int current_jobsize = jobsize;
	    int thread_num = omp_get_thread_num();
        
        // Private device memory
        cu::DeviceMemory d_uvw(UVW_SIZE);
    	cu::DeviceMemory d_subgrid(SUBGRID_SIZE);
    
	    for (int bl = thread_num * jobsize; bl < NR_BASELINES; bl += nr_streams * jobsize) {
	        // Prevent overflow
		    current_jobsize = bl + jobsize > NR_BASELINES ? NR_BASELINES - bl : jobsize;
           
		    // Number of elements in batch
		    size_t uvw_elements     = NR_TIME * 3;
		    size_t subgrid_elements = SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS;
		
		    // Pointers to data for current batch
		    void *uvw_ptr     = (float *) h_uvw + bl * uvw_elements;
		    void *subgrid_ptr = (float complex *) h_subgrid + bl * subgrid_elements;

	        // Copy input to device
            records[0].enqueue(htodstream);
            #if INPUT
            htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, UVW_SIZE);
	        #endif
            records[1].enqueue(htodstream);

            // Wait for memory transfer to finish
            htodstream.record(event);
            executestream.waitEvent(event);

	        // Launch split kernel
            records[2].enqueue(executestream);
            #if SPLITTER
	        kernel_splitter.launchAsync(
	            executestream, current_jobsize, bl, d_uvw, d_subgrid, d_grid);
	        #endif
            records[3].enqueue(executestream);
            
	        // Wait for computation of sugrid to finish
	        executestream.record(event);
	        dtohstream.waitEvent(event);
            
            // Copy subgrid to host
	        records[4].enqueue(dtohstream);
            #if OUTPUT
	        dtohstream.memcpyDtoHAsync(subgrid_ptr, d_subgrid, SUBGRID_SIZE);
            #endif
            records[5].enqueue(dtohstream);

            // Wait for memory transfers to finish
            dtohstream.record(event);
            event.synchronize();  
            
		    #if REPORT_VERBOSE
		    #if INPUT
		    report("   input", records[0], records[1], 0, UVW_SIZE);
		    #endif
		    #if SPLITTER
		    report("splitter", records[2], records[3],
		                       kernel_splitter.flops(current_jobsize),
		                       kernel_splitter.bytes(current_jobsize));
		    #endif
		    #if OUTPUT
		    report("  output", records[4], records[5], 0, SUBGRID_SIZE);
		    #endif
		    #endif
		    
		    // Update total runtime
		    total_time_in[thread_num]       += runtime(records[0], records[1]);
		    total_time_splitter[thread_num] += runtime(records[2], records[3]);
		    total_time_out[thread_num]      += runtime(records[4], records[5]);
            total_bytes_in[thread_num]      += UVW_SIZE;
		    total_bytes_out[thread_num]     += SUBGRID_SIZE;
		    total_jobs[thread_num]          += jobsize;
	    }
	}

    // Measure total runtime
    records_total[1].enqueue(dtohstream);
    dtohstream.synchronize();

    #if REPORT_TOTAL
    std::clog << std::endl;
    
    // Report performance per stream
    for (int t = 0; t < nr_streams; t++) {
        std::clog << "--- stream " << t << " ---" << std::endl;
        int jobsize = total_jobs[t];
        report("   input", total_time_in[t], 0, total_bytes_in[t]);
        report("splitter", total_time_splitter[t],
                          kernel_splitter.flops(jobsize),
                          kernel_splitter.bytes(jobsize));
        report("  output", total_time_out[t], 0, total_bytes_out[t]);
        std::clog << std::endl;
    }
    
    // Report overall performance
    std::clog << "--- overall ---" << std::endl;
    report("  input", records_input[0], records_input[1], 0, sizeof(GridType));
    report("  total", records_total[0], records_total[1], 0, 0);
    double total_runtime = runtime(records_total[0], records_total[1]);
    report_subgrids(total_runtime);
    std::clog << std::endl;
    #endif
}


/*
	Degridder
*/
void callback_degridder_input(CUstream, CUresult, void *userData) {
    State *s = (State *) userData;
    int current_jobsize = s->jobsize;
    s->runtime = runtime(s->startRecord, s->stopRecord);
    s->flops = 0;
    s->bytes = SUBGRID_SIZE;
    #if MEASURE_POWER
    s->power = power(s->startRecord, s->stopRecord);
    #endif
    #if REPORT_VERBOSE
    report("    input", s->startRecord, s->stopRecord, s->flops, s->bytes);
    #endif
}

void callback_degridder_degridder(CUstream, CUresult, void *userData) {
     State *s = (State *) userData;
    int current_jobsize = s->jobsize;
    s->runtime = runtime(s->startRecord, s->stopRecord);
    s->flops = KernelDegridder::flops(current_jobsize);
    s->bytes = KernelDegridder::bytes(current_jobsize);
    #if MEASURE_POWER
    s->power = power(s->startRecord, s->stopRecord);
    #endif
    #if REPORT_VERBOSE
    report("degridder", s->startRecord, s->stopRecord, s->flops, s->bytes);
    #endif
}

void callback_degridder_fft(CUstream, CUresult, void *userData) {
     State *s = (State *) userData;
    int current_jobsize = s->jobsize;
    s->runtime = runtime(s->startRecord, s->stopRecord);
    s->flops = KernelFFT::flops(SUBGRIDSIZE, current_jobsize);
    s->bytes = KernelFFT::bytes(SUBGRIDSIZE, current_jobsize);
    #if MEASURE_POWER
    s->power = power(s->startRecord, s->stopRecord);
    #endif
    #if REPORT_VERBOSE
    report("      fft", s->startRecord, s->stopRecord, s->flops, s->bytes);
    #endif
}

void callback_degridder_output(CUstream, CUresult, void *userData) {
    State *s = (State *) userData;
    int current_jobsize = s->jobsize;
    s->runtime = runtime(s->startRecord, s->stopRecord);
    s->flops = 0;
    s->bytes = UVW_SIZE + VISIBILITY_SIZE;
    #if MEASURE_POWER
    s->power = power(s->startRecord, s->stopRecord);
    #endif
    #if REPORT_VERBOSE
    report("   output", s->startRecord, s->stopRecord, s->flops, s->bytes);
    #endif
}

void run_degridder(
    cu::Context &context, int nr_streams, int jobsize,
    cu::HostMemory &h_visibilities, cu::HostMemory &h_uvw,
    cu::HostMemory &h_subgrid, cu::DeviceMemory &d_wavenumbers,
    cu::DeviceMemory &d_spheroidal, cu::DeviceMemory &d_aterm,
    cu::DeviceMemory &d_baselines) {
    
    // Initialize
    Record records_total[2];
    cu::Stream executestream;
    cu::Stream htodstream;
    cu::Stream dtohstream;
	
	// Load kernel module
	cu::Module module_degridder(PTX_DEGRIDDER);
	
	// Load kernel function
	KernelDegridder kernel_degridder(module_degridder, KERNEL_DEGRIDDER);
	
    // Timing variables
	double total_time_degridder[nr_streams];
	double total_time_fft[nr_streams];
	double total_time_input[nr_streams];
	double total_time_output[nr_streams];
    double total_bytes_input[nr_streams];
    double total_bytes_output[nr_streams];
    long total_jobs[nr_streams];
    for (int t = 0; t < nr_streams; t++) {
        total_time_degridder[t] = 0;
        total_time_fft[t]       = 0;
        total_time_input[t]     = 0;
        total_time_output[t]    = 0;
        total_bytes_input[t]    = 0;
        total_bytes_output[t]   = 0;
        total_jobs[t]           = 0;
    }

    // Number of iterations per stream
    int nr_iterations = ((NR_BASELINES / nr_streams) / jobsize) + 1;
    
    // Start degridder
    records_total[0].enqueue(executestream);
    #pragma omp parallel num_threads(nr_streams)
	{
	    // Initialize
	    context.setCurrent();
        cu::Event executeFinished;
        cu::Event inputFree;
	    int current_jobsize = jobsize;
        int thread_num = omp_get_thread_num();
        KernelFFT kernel_fft;
        int iteration = 0;
       
        // Initialize states 
        State states_input[nr_iterations];
        State states_degridder[nr_iterations];
        State states_fft[nr_iterations];
        State states_output[nr_iterations];
    
	    // Private memory
    	cu::DeviceMemory d_uvw(UVW_SIZE);
    	cu::DeviceMemory d_visibilities(VISIBILITY_SIZE);
    	cu::DeviceMemory d_subgrid(SUBGRID_SIZE);

	    for (int bl = thread_num * jobsize; bl < NR_BASELINES; bl += nr_streams * jobsize) {
	        // Prevent overflow
		    current_jobsize = bl + jobsize > NR_BASELINES ? NR_BASELINES - bl : jobsize;
    
            // States
            State &state_input      = states_input[iteration];
            State &state_degridder  = states_degridder[iteration];
            State &state_fft        = states_fft[iteration];
            State &state_output     = states_output[iteration];
     
            // Set jobsize in states 
            state_input.jobsize     = current_jobsize;
            state_degridder.jobsize = current_jobsize;
            state_fft.jobsize       = current_jobsize;
            state_output.jobsize    = current_jobsize;

		    // Number of elements in batch
		    size_t uvw_elements        = NR_TIME * 3;
		    size_t visibility_elements = NR_TIME * NR_CHANNELS * NR_POLARIZATIONS;
		    size_t subgrid_elements     = SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS;
		
		    // Pointers to data for current batch
		    void *visibilities_ptr = (float complex *) h_visibilities + bl * visibility_elements;
		    void *uvw_ptr          = (float *) h_uvw + bl * uvw_elements;
		    void *subgrid_ptr      = (float complex *) h_subgrid + bl * subgrid_elements;

            // Wait for previous computation to finish
            executestream.record(executeFinished);
            htodstream.waitEvent(executeFinished);
		
	        // Copy input data to device
            #pragma omp critical
            {
                inputFree.synchronize();
                htodstream.waitEvent(inputFree);
                state_input.startRecord.enqueue(htodstream);
                #if INPUT
	            htodstream.memcpyHtoDAsync(d_uvw, uvw_ptr, UVW_SIZE);
	            htodstream.memcpyHtoDAsync(d_subgrid, subgrid_ptr, SUBGRID_SIZE);
                #endif
	            state_input.stopRecord.enqueue(htodstream);
                htodstream.addCallback(&callback_degridder_input, &state_input);
            }

            // Create FFT plan
            #if FFT
            #if ORDER == ORDER_BL_V_U_P
            htodstream.plan(SUBGRIDSIZE, current_jobsize, FFT_LAYOUT_YXP);
            #elif ORDER == ORDER_BL_P_V_U
            kernel_fft.plan(SUBGRIDSIZE, current_jobsize, FFT_LAYOUT_PYX);
            #endif
            #endif
	        
            #pragma omp critical
            {
    	        // Launch FFT
                executestream.waitEvent(state_input.stopRecord.event);
    	        state_fft.startRecord.enqueue(executestream);
                #if FFT
    	        kernel_fft.launchAsync(executestream, d_subgrid, CUFFT_INVERSE);
                #endif
    	        state_fft.stopRecord.enqueue(executestream);
                executestream.addCallback(&callback_degridder_fft, &state_fft);
    	        
    	        // Launch degridder kernel
    	        state_degridder.startRecord.enqueue(executestream);
                #if DEGRIDDER
    	        kernel_degridder.launchAsync(
    	            executestream, current_jobsize, bl,
    		        d_subgrid, d_uvw, d_wavenumbers, d_aterm,
                    d_baselines, d_spheroidal, d_visibilities);
                #endif
    	        state_degridder.stopRecord.enqueue(executestream);
                executestream.record(inputFree);
                executestream.addCallback(&callback_degridder_degridder, &state_degridder);
            }
                
            #pragma omp critical
            {
                // Copy visibilities to host
                dtohstream.waitEvent(state_degridder.stopRecord.event);
	            state_output.startRecord.enqueue(dtohstream);
                #if OUTPUT
	            dtohstream.memcpyDtoHAsync(visibilities_ptr, d_visibilities, VISIBILITY_SIZE);
                #endif
	            state_output.stopRecord.enqueue(dtohstream);
                dtohstream.addCallback(&callback_degridder_output, &state_output);
            }
			
            // Go to next iteration
            total_jobs[thread_num] += current_jobsize;
            iteration++;
	    }
        
        // Sum totals
        #if REPORT_TOTAL
        for (int i = 0; i < nr_iterations; i++) {
            total_time_degridder[thread_num] += states_degridder[i].runtime;
            total_time_fft[thread_num]       += states_fft[i].runtime;
            total_time_input[thread_num]     += states_input[i].runtime;
            total_time_output[thread_num]    += states_output[i].runtime;
            total_bytes_input[thread_num]    += states_input[i].bytes;
            total_bytes_output[thread_num]   += states_output[i].bytes;
        }
        #endif
 
        // Wait for final transfers to finish
        dtohstream.synchronize();
    }
    
    // Measure total runtime
    records_total[1].enqueue(executestream);
    executestream.synchronize();

    #if REPORT_TOTAL
    std::clog << std::endl;
    KernelFFT kernel_fft;
    // Print report per thread
    for (int t = 0; t < nr_streams; t++) {
        jobsize = total_jobs[t];
        std::clog << "--- stream " << t << " ---" << std::endl;
        report("      fft", total_time_fft[t],
                            kernel_fft.flops(SUBGRIDSIZE, jobsize),
                            kernel_fft.bytes(SUBGRIDSIZE, jobsize));
        report("degridder", total_time_degridder[t],
                            kernel_degridder.flops(jobsize),
                            kernel_degridder.bytes(jobsize));
        report("    input", total_time_input[t], 0,
                            total_bytes_input[t]);
        report("   output", total_time_output[t], 0,
                            total_bytes_output[t]);
        std::clog << std::endl;
    }
    
    // Print overall report
    std::clog << "--- overall ---" << std::endl;
    long total_flops = kernel_degridder.flops(NR_BASELINES) +
                       kernel_fft.flops(SUBGRIDSIZE, NR_BASELINES);
    report("    total", records_total[0], records_total[1], total_flops, 0);
    double total_runtime = runtime(records_total[0], records_total[1]);
    report_visibilities(total_runtime);
    std::clog << std::endl;
    #endif
}


/*
	FFT
*/
void run_fft(
    cu::Context &context,
    cu::HostMemory &h_grid,
    int sign) {
    
    // Initialize
    context.setCurrent();
    cu::Stream stream;
    Record records[6];
    Record records_total[2];
    
	// Load kernel function
	KernelFFT kernel_fft;
	
	// Start measure total runtime
	records_total[0].enqueue(stream);
	
    // Create FFT plan
    #if FFT
    kernel_fft.plan(GRIDSIZE, 1, FFT_LAYOUT_YXP);
    #endif
    
    // Copy grid to device
	cu::DeviceMemory d_grid(sizeof(GridType));
	records[0].enqueue(stream);
	stream.memcpyHtoDAsync(d_grid, h_grid);
	records[1].enqueue(stream);

    // Launch FFT
    records[2].enqueue(stream);
    #if FFT
    kernel_fft.launchAsync(stream, d_grid, sign);
    #endif
    records[3].enqueue(stream);
    
    // Copy grid to host
    records[4].enqueue(stream);
    stream.memcpyDtoHAsync(h_grid, d_grid, sizeof(GridType));
    records[5].enqueue(stream);
    
    // Wait for computation to finish
    records_total[1].enqueue(stream);
    stream.synchronize();
    
    #if REPORT_TOTAL
    report(" input", records[0], records[1], 0,
        sizeof(GridType));
    report("   fft", records[2], records[3],
        kernel_fft.flops(GRIDSIZE, 1),
        kernel_fft.bytes(GRIDSIZE, 1));
    report("output", records[4], records[5], 0,
        sizeof(GridType));
    report(" total", records_total[0], records_total[1], 0, 0);
    std::clog << std::endl;
    #endif
}

}
