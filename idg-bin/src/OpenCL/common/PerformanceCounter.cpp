#include "PerformanceCounter.h"

using namespace idg;

/*
    Utility
*/
double PerformanceCounter::get_runtime(cl_event event) {
    cl_ulong start, end;
    double runtime = 0;
    if (clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL) == CL_SUCCESS &&
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL) == CL_SUCCESS) {
        runtime = (end - start) * 1e-9;
    }
    return runtime;
}

double PerformanceCounter::get_runtime(cl_event event1, cl_event event2) {
    cl_ulong start, end;
    double runtime = 0;
    if (clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL) == CL_SUCCESS &&
        clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL) == CL_SUCCESS) {
        runtime = (end - start) * 1e-9;
    }
    return runtime;
}

/*
    Performance Counter for one event
*/
void PerformanceCounter::startPowerMeasurement(cl_event event, cl_int, void *user_data) {
    Descriptor *descriptor = static_cast<Descriptor *>(user_data);
    descriptor->startState = descriptor->powerSensor->read();
}

void PerformanceCounter::stopPowerMeasurement(cl_event event, cl_int, void *user_data) {
    Descriptor *descriptor = static_cast<Descriptor *>(user_data);
    descriptor->stopState = descriptor->powerSensor->read();
}

void PerformanceCounter::report(cl_event event, cl_int, void *user_data) {
    Descriptor *descriptor = static_cast<Descriptor *>(user_data);
    PowerSensor *powerSensor = descriptor->powerSensor;
    double watts = powerSensor->Watt(descriptor->startState, descriptor->stopState);
    double runtime = get_runtime(event);
    auxiliary::report(descriptor->name, runtime, descriptor->flops, descriptor->bytes, watts);
    delete descriptor;
}

void PerformanceCounter::doOperation(cl::Event &event, const char *name, uint64_t flops, uint64_t bytes) {
    Descriptor *descriptor = new Descriptor;
    descriptor->name = name;
    descriptor->flops = flops;
    descriptor->bytes = bytes;
    descriptor->powerSensor = powerSensor;
    event.setCallback(CL_SUBMITTED, &PerformanceCounter::startPowerMeasurement, descriptor);
    event.setCallback(CL_RUNNING, &PerformanceCounter::stopPowerMeasurement, descriptor);
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
    descriptor->stopState = descriptor->powerSensor->read();
    descriptor->runtime += get_runtime(event);
}

void PerformanceCounter::report2(cl_event event, cl_int, void *user_data) {
    Descriptor *descriptor = static_cast<Descriptor *>(user_data);
    PowerSensor *powerSensor = descriptor->powerSensor;
    double watts = powerSensor->Watt(descriptor->startState, descriptor->stopState);
    auxiliary::report(descriptor->name, descriptor->runtime, descriptor->flops, descriptor->bytes, watts);
    delete descriptor;
}

void PerformanceCounter::doOperation(cl::Event &start, cl::Event &end, const char *name, uint64_t flops, uint64_t bytes) {
    Descriptor *descriptor = new Descriptor;
    descriptor->name = name;
    descriptor->flops = flops;
    descriptor->bytes = bytes;
    descriptor->runtime = 0;
    descriptor->powerSensor = powerSensor;
    start.setCallback(CL_SUBMITTED, &PerformanceCounter::startPowerMeasurement, descriptor);
    start.setCallback(CL_RUNNING, &PerformanceCounter::stopTimingMeasurement, descriptor);
    end.setCallback(CL_RUNNING, &PerformanceCounter::stopPowerAndTimingMeasurement, descriptor);
    end.setCallback(CL_COMPLETE, &PerformanceCounter::report2, descriptor);
}

void PerformanceCounter::setPowerSensor(PowerSensor *_powerSensor) {
    powerSensor = _powerSensor;
}
