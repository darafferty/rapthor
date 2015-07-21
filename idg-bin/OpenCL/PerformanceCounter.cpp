#include "PerformanceCounter.h"

PerformanceCounter::PerformanceCounter(const char *name) :
    name(name),
    total_flops(0),
    total_bytes(0),
    total_runtime(0),
    nr_callbacks(0) {}

void PerformanceCounter::doOperation(cl::Event &event, uint64_t flops, uint64_t bytes) {
    usleep(1000);
    #pragma omp atomic
    nr_callbacks += 1;
    callback = [=] (cl_event _event) {
            cl_ulong start, end;
                if (clGetEventProfilingInfo(_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL) == CL_SUCCESS &&
                    clGetEventProfilingInfo(_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL) == CL_SUCCESS) {
                    double runtime = (end - start) * 1e-9;
                    #pragma omp atomic
                    total_runtime += runtime;
                    #if REPORT_VERBOSE
                    report(name, runtime, flops, bytes);
                    #endif
                    #pragma omp atomic
                    nr_callbacks -= 1;
                }
        };
        event.setCallback(CL_COMPLETE, &PerformanceCounter::eventCompleteCallBack, this);
        #pragma omp atomic
        total_flops += flops;
        #pragma omp atomic
        total_bytes += bytes;
}

void PerformanceCounter::eventCompleteCallBack(cl_event event, cl_int, void *user_data) {
    static_cast<PerformanceCounter *>(user_data)->callback(event);
}

void PerformanceCounter::report(const char *name, double runtime, uint64_t flops, uint64_t bytes) {
	#pragma omp critical(clog)
	{
    std::clog << std::setw(14) << name;
    std::clog << std::setprecision(5) << std::fixed;
    std::clog << ": " << runtime << " s";
    if (flops != 0)
		std::clog << ", " << flops / runtime * 1e-12 << " TFLOPS";
    if (bytes != 0)
		std::clog << ", " << bytes / runtime * 1e-9 << " GB/s";
    std::clog << std::endl;
	}
}

void PerformanceCounter::report_total() {
    #if REPORT_TOTAL
    report(name, total_runtime, total_flops, total_bytes);
    #endif
}

void PerformanceCounter::wait() {
    while (nr_callbacks > 0) {
        //std::clog << name << " " << nr_callbacks << std::endl;
        usleep(1000);
    }
}
