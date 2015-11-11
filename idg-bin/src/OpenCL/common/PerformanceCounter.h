#include <iostream>
#include <iomanip>
#include <functional>
#include <unistd.h>

#include <CL/cl.hpp>

#include "auxiliary.h"
#include "PowerSensor.h"

class PerformanceCounter {
    struct Descriptor
    {
        const char *name;
        uint64_t flops;
        uint64_t bytes;
        #if defined(MEASURE_POWER_ARDUINO)
        PowerSensor *powerSensor;
        PowerSensor::State startState, stopState;
        #endif
    };

    public:
        void doOperation(cl::Event &event, const char *name, uint64_t flops, uint64_t bytes);
        #if defined(MEASURE_POWER_ARDUINO)
        PowerSensor *powerSensor;
        void setPowerSensor(PowerSensor *_powerSensor);
        #endif

   private:
        static void startPowerMeasurement(cl_event event, cl_int, void *user_data);
        static void stopPowerMeasurement(cl_event event, cl_int, void *user_data);
        static void report(cl_event, cl_int, void *user_data);
};
