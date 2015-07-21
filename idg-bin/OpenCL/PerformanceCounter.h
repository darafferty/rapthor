#include <iostream>
#include <iomanip>
#include <functional>
#include <unistd.h>

#include <CL/cl.hpp>

#define REPORT_VERBOSE 1
#define REPORT_TOTAL   1


class PerformanceCounter {
    public:
        PerformanceCounter(const char *name);
        void doOperation(cl::Event &event, uint64_t flops, uint64_t bytes);
        static void report(const char *name, double runtime, uint64_t flops, uint64_t bytes);
        void report_total();
        void wait();

    private:
        static void eventCompleteCallBack(cl_event, cl_int, void *counter);

    public:
       const char *name;
       uint64_t total_flops;
       uint64_t total_bytes;
       double   total_runtime;
       volatile int nr_callbacks;

   private:    
       std::function<void (cl_event)> callback;
};
