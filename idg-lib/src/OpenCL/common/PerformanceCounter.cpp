#include "PerformanceCounter.h"

using namespace idg;

PerformanceCounter::PerformanceCounter(const char *name) :
    name(name) {}

void PerformanceCounter::doOperation(uint64_t flops, uint64_t bytes) {
    #if defined(MEASURE_POWER)
    powerStates[0] = powerSensor->read();
    #endif
    callback = [=] (cl_event _event) {
            cl_ulong start, end;
                if (clGetEventProfilingInfo(_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL) == CL_SUCCESS &&
                    clGetEventProfilingInfo(_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL) == CL_SUCCESS) {
                    double runtime = (end - start) * 1e-9;
                    #if defined(MEASURE_POWER)
                    powerStates[1] = powerSensor->read();
                    double watts = PowerSensor::Watt(powerStates[0], powerStates[1]);
                    auxiliary::report(name, runtime, flops, bytes, watts);
                    #else
                    auxiliary::report(name, runtime, flops, bytes, 0);
                    #endif
                }
        };
        event.setCallback(CL_COMPLETE, &PerformanceCounter::eventCompleteCallBack, this);
}

void PerformanceCounter::eventCompleteCallBack(cl_event event, cl_int, void *user_data) {
    static_cast<PerformanceCounter *>(user_data)->callback(event);
}

#if defined(MEASURE_POWER)
void PerformanceCounter::setPowerSensor(PowerSensor *_powerSensor) {
    powerSensor = _powerSensor;
}
#endif
