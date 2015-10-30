#include <iostream>
#include <iomanip>
#include <functional>
#include <unistd.h>

#include <CL/cl.hpp>

#include "auxiliary.h"
#include "PowerSensor.h"

class PerformanceCounter {
    public:
        PerformanceCounter(const char *name);
        void doOperation(uint64_t flops, uint64_t bytes);
        static void report(const char *name, double runtime, uint64_t flops, uint64_t bytes);

    private:
        static void eventSubmittedCallBack(cl_event, cl_int, void *counter);
        static void eventRunningCallBack(cl_event, cl_int, void *counter);
        static void eventCompleteCallBack(cl_event, cl_int, void *counter);
        #if defined(MEASURE_POWER_ARDUINO)
        PowerSensor::State powerStates[2];
        #endif

    public:
        const char *name;
        cl::Event event;
        #if defined(MEASURE_POWER_ARDUINO)
        PowerSensor *powerSensor;
        void setPowerSensor(PowerSensor *_powerSensor);
        #endif

   private:
       std::function<void (cl_event)> callback_submitted;
       std::function<void (cl_event)> callback_running;
       std::function<void (cl_event)> callback_completed;
};
