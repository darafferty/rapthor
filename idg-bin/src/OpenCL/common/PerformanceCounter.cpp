#include "PerformanceCounter.h"

using namespace idg;

/*
    Utility
*/
double get_runtime(cl_event event) {
    cl_ulong start, end;
    double runtime = 0;
    if (clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL) == CL_SUCCESS &&
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL) == CL_SUCCESS) {
        runtime = (end - start) * 1e-9;
    }
    return runtime;
}


/*
    Performance Counter for one event
*/
void PerformanceCounter::startPowerMeasurement(cl_event event, cl_int, void *user_data) {
    #if defined(MEASURE_POWER_ARDUINO)
    Descriptor *descriptor = static_cast<Descriptor *>(user_data);
    descriptor->startState = descriptor->powerSensor->read();
    #endif
}

void PerformanceCounter::stopPowerMeasurement(cl_event event, cl_int, void *user_data) {
    #if defined(MEASURE_POWER_ARDUINO)
    Descriptor *descriptor = static_cast<Descriptor *>(user_data);
    descriptor->stopState = descriptor->powerSensor->read();
    #endif
}

void PerformanceCounter::report(cl_event event, cl_int, void *user_data) {
    Descriptor *descriptor = static_cast<Descriptor *>(user_data);
    double watts = 0;
    #if defined(MEASURE_POWER_ARDUINO)
    watts = PowerSensor::Watt(descriptor->startState, descriptor->stopState);
    #endif
    double runtime = get_runtime(event);
    auxiliary::report(descriptor->name, runtime, descriptor->flops, descriptor->bytes, watts);
    delete descriptor;
}

void PerformanceCounter::doOperation(cl::Event &event, const char *name, uint64_t flops, uint64_t bytes) {
    Descriptor *descriptor = new Descriptor;
    descriptor->name = name;
    descriptor->flops = flops;
    descriptor->bytes = bytes;
    #if defined(MEASURE_POWER_ARDUINO)
    descriptor->powerSensor = powerSensor;
    event.setCallback(CL_SUBMITTED, &PerformanceCounter::startPowerMeasurement, descriptor);
    event.setCallback(CL_RUNNING, &PerformanceCounter::stopPowerMeasurement, descriptor);
    #endif
    event.setCallback(CL_COMPLETE, &PerformanceCounter::report, descriptor);
}


/*
    Performance Counter for two events
*/
void PerformanceCounter::stopTimingMeasurement(cl_event event, cl_int, void *user_data) {
    Descriptor *descriptor = static_cast<Descriptor *>(user_data);
    descriptor->runtime += get_runtime(event);
}

void PerformanceCounter::stopPowerAndTimingMeasurement(cl_event event, cl_int, void *user_data) {
    Descriptor *descriptor = static_cast<Descriptor *>(user_data);
    #if defined(MEASURE_POWER_ARDUINO)
    descriptor->stopState = descriptor->powerSensor->read();
    #endif
    descriptor->runtime += get_runtime(event);
}

void PerformanceCounter::report2(cl_event event, cl_int, void *user_data) {
    Descriptor *descriptor = static_cast<Descriptor *>(user_data);
    double watts = 0;
    #if defined(MEASURE_POWER_ARDUINO)
    watts = PowerSensor::Watt(descriptor->startState, descriptor->stopState);
    #endif
    auxiliary::report(descriptor->name, descriptor->runtime, descriptor->flops, descriptor->bytes, watts);
    delete descriptor;
}

void PerformanceCounter::doOperation(cl::Event &start, cl::Event &end, const char *name, uint64_t flops, uint64_t bytes) {
    Descriptor *descriptor = new Descriptor;
    descriptor->name = name;
    descriptor->flops = flops;
    descriptor->bytes = bytes;
    descriptor->runtime = 0;
    #if defined(MEASURE_POWER_ARDUINO)
    descriptor->powerSensor = powerSensor;
    start.setCallback(CL_SUBMITTED, &PerformanceCounter::startPowerMeasurement, descriptor);
    #endif
    start.setCallback(CL_RUNNING, &PerformanceCounter::stopTimingMeasurement, descriptor);
    end.setCallback(CL_RUNNING, &PerformanceCounter::stopPowerAndTimingMeasurement, descriptor);
    end.setCallback(CL_COMPLETE, &PerformanceCounter::report2, descriptor);
}

#if defined(MEASURE_POWER_ARDUINO)
void PerformanceCounter::setPowerSensor(PowerSensor *_powerSensor) {
    powerSensor = _powerSensor;
}
#endif
