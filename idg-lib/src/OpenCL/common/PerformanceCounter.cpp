#include "PerformanceCounter.h"

using namespace idg;

PerformanceCounter::PerformanceCounter(const char *name) :
    name(name) {}

void PerformanceCounter::doOperation(uint64_t flops, uint64_t bytes) {
    #if defined(MEASURE_POWER_ARDUINO)
    // Set power measurement callbacks
    callback_submitted = [=] (cl_event _event) {
        powerStates[0] = powerSensor->read();
    };
    callback_running = [=] (cl_event _event) {
        powerStates[1] = powerSensor->read();
    };
    event.setCallback(CL_SUBMITTED, &PerformanceCounter::eventSubmittedCallBack, this);
    event.setCallback(CL_RUNNING, &PerformanceCounter::eventRunningCallBack, this);
    #endif
    // Set performance reporting callback
    callback_completed = [=] (cl_event _event) {
        double runtime = 0;
        double watts = 0;
        #if defined(MEASURE_POWER_ARDUINO) 
        runtime = PowerSensor::seconds(powerStates[0], powerStates[1]);
        watts = PowerSensor::Watt(powerStates[0], powerStates[1]);
        #else
        cl_ulong start, end;
        if (clGetEventProfilingInfo(_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL) == CL_SUCCESS &&
            clGetEventProfilingInfo(_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL) == CL_SUCCESS) {
            runtime = (end - start) * 1e-9;
        }
        watts = 0;
        #endif
        auxiliary::report(name, runtime, flops, bytes, watts);
    };
    event.setCallback(CL_COMPLETE, &PerformanceCounter::eventCompleteCallBack, this);
}

void PerformanceCounter::eventSubmittedCallBack(cl_event event, cl_int, void *user_data) {
    static_cast<PerformanceCounter *>(user_data)->callback_submitted(event);
}

void PerformanceCounter::eventRunningCallBack(cl_event event, cl_int, void *user_data) {
    static_cast<PerformanceCounter *>(user_data)->callback_running(event);
}

void PerformanceCounter::eventCompleteCallBack(cl_event event, cl_int, void *user_data) {
    static_cast<PerformanceCounter *>(user_data)->callback_completed(event);
}

#if defined(MEASURE_POWER_ARDUINO)
void PerformanceCounter::setPowerSensor(PowerSensor *_powerSensor) {
    powerSensor = _powerSensor;
}
#endif
