#include "PerformanceCounter.h"

namespace idg {

    namespace proxy {
        PerformanceCounter::PerformanceCounter(const char *name) :
            name(name) {}

        void PerformanceCounter::doOperation(cl::Event &event, uint64_t flops, uint64_t bytes) {
            usleep(1000);
            callback = [=] (cl_event _event) {
                    cl_ulong start, end;
                        if (clGetEventProfilingInfo(_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL) == CL_SUCCESS &&
                            clGetEventProfilingInfo(_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL) == CL_SUCCESS) {
                            double runtime = (end - start) * 1e-9;
                            auxiliary::report(name, runtime, flops, bytes, 0);
                        }
                };
                event.setCallback(CL_COMPLETE, &PerformanceCounter::eventCompleteCallBack, this);
        }

        void PerformanceCounter::eventCompleteCallBack(cl_event event, cl_int, void *user_data) {
            static_cast<PerformanceCounter *>(user_data)->callback(event);
        }
    }
}
