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
void PerformanceCounter::startMeasurement(cl_event event, cl_int, void *user_data) {
    Descriptor *descriptor = static_cast<Descriptor *>(user_data);
    descriptor->startState = descriptor->powerSensor->read();
}

void PerformanceCounter::stopMeasurement(cl_event event, cl_int, void *user_data) {
    Descriptor *descriptor = static_cast<Descriptor *>(user_data);
    PowerSensor *powerSensor = descriptor->powerSensor;
    descriptor->stopState = descriptor->powerSensor->read();
    auxiliary::report(descriptor->name, descriptor->flops, descriptor->bytes,
                      powerSensor, descriptor->startState, descriptor->stopState);
    delete descriptor;
}

void PerformanceCounter::doOperation(cl::Event &event, const char *name, uint64_t flops, uint64_t bytes) {
    Descriptor *descriptor = new Descriptor;
    descriptor->name  = name;
    descriptor->flops = flops;
    descriptor->bytes = bytes;
    descriptor->powerSensor = powerSensor;
    event.setCallback(CL_SUBMITTED, &PerformanceCounter::startMeasurement, descriptor);
    event.setCallback(CL_COMPLETE, &PerformanceCounter::stopMeasurement, descriptor);
}


/*
    Performance Counter for two events
*/
void PerformanceCounter::stopMeasurement2(cl_event event, cl_int, void *user_data) {
    Descriptor *descriptor = static_cast<Descriptor *>(user_data);
    descriptor->stopState = descriptor->powerSensor->read();
    PowerSensor *powerSensor = descriptor->powerSensor;
    auxiliary::report(descriptor->name, descriptor->flops, descriptor->bytes,
                      powerSensor, descriptor->startState, descriptor->stopState);
    delete descriptor;
}

void PerformanceCounter::doOperation(cl::Event &start, cl::Event &end, const char *name, uint64_t flops, uint64_t bytes) {
    Descriptor *descriptor = new Descriptor;
    descriptor->name  = name;
    descriptor->flops = flops;
    descriptor->bytes = bytes;
    descriptor->powerSensor = powerSensor;
    start.setCallback(CL_SUBMITTED, &PerformanceCounter::startMeasurement, descriptor);
    end.setCallback(CL_RUNNING, &PerformanceCounter::stopMeasurement2, descriptor);
}


/*
    Common
*/
void PerformanceCounter::setPowerSensor(PowerSensor *_powerSensor) {
    powerSensor = _powerSensor;
}
