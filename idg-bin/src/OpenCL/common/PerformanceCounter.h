#include <iostream>
#include <iomanip>
#include <functional>
#include <unistd.h>

#include <CL/cl.hpp>

#include "auxiliary.h"

class PerformanceCounter {
    public:
        PerformanceCounter(const char *name);
        void doOperation(uint64_t flops, uint64_t bytes);
        static void report(const char *name, double runtime, uint64_t flops, uint64_t bytes);

    private:
        static void eventCompleteCallBack(cl_event, cl_int, void *counter);

    public:
        const char *name;
        cl::Event event;

   private:
       std::function<void (cl_event)> callback;
};
