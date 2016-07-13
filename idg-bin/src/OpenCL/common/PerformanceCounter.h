#ifndef IDG_OPENCL_PERFORMANCECOUNTER_H_
#define IDG_OPENCL_PERFORMANCECOUNTER_H_

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <iostream>
#include <iomanip>
#include <functional>
#include <unistd.h>

#include "idg-common.h"
#include "idg-powersensor.h"

class PerformanceCounter {
    struct Descriptor
    {
        const char *name;
        uint64_t flops;
        uint64_t bytes;
        double runtime;

        PowerSensor *powerSensor;
        PowerSensor::State startState, stopState;
    };

    public:
        void doOperation(cl::Event &event, const char *name, uint64_t flops, uint64_t bytes);
        void doOperation(cl::Event &tart, cl::Event &end, const char *name, uint64_t flops, uint64_t bytes);

        PowerSensor *powerSensor;
        void setPowerSensor(PowerSensor *_powerSensor);

        static double get_runtime(cl_event event);
        static double get_runtime(cl_event event1, cl_event event2);

   private:
        static void startPowerMeasurement(cl_event event, cl_int, void *user_data);
        static void stopPowerMeasurement(cl_event event, cl_int, void *user_data);
        static void stopTimingMeasurement(cl_event event, cl_int, void *user_data);
        static void stopPowerAndTimingMeasurement(cl_event event, cl_int, void *user_data);
        static void report(cl_event, cl_int, void *user_data);
        static void report2(cl_event, cl_int, void *user_data);
};

#endif
