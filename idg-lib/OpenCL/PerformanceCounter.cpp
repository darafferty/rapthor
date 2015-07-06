#include "PerformanceCounter.h"

PerformanceCounter::PerformanceCounter(const char *name) :
    name(name),
    total_flops(0),
    total_bytes(0),
    total_runtime(0) {}

PerformanceCounter::~PerformanceCounter() {
    #if REPORT_TOTAL
    report(total_runtime, total_flops, total_bytes);
    #endif
}

void PerformanceCounter::doOperation(cl::Event &event, uint64_t flops, uint64_t bytes) {
    event.setCallback(CL_COMPLETE, &PerformanceCounter::eventCompleteCallBack, this);
    _flops = flops;
    _bytes = bytes;
    total_flops += flops;
    total_bytes += bytes;
}

void PerformanceCounter::eventCompleteCallBack(cl_event event, cl_int, void *counter) {
    cl_ulong start, end;
    if (clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL) == CL_SUCCESS &&
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL) == CL_SUCCESS) {
        double runtime = (end - start) * 1e-9;
        static_cast<PerformanceCounter *>(counter)->total_runtime += runtime;
        #if REPORT_VERBOSE
        //report(runtime, _flops, _bytes);
        #endif
    }
}

void PerformanceCounter::report(double runtime, uint64_t flops, uint64_t bytes) {
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
