#include "PerformanceCounter.h"

using namespace idg;

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
    double runtime = 0;
    cl_ulong start, end;
    if (clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL) == CL_SUCCESS &&
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL) == CL_SUCCESS) {
        runtime = (end - start) * 1e-9;
    }
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

#if defined(MEASURE_POWER_ARDUINO)
void PerformanceCounter::setPowerSensor(PowerSensor *_powerSensor) {
    powerSensor = _powerSensor;
}
#endif
