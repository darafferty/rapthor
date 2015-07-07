#include <iostream>
#include <iomanip>
#include <functional>

#include <CL/cl.hpp>

#define REPORT_VERBOSE 1
#define REPORT_TOTAL   1


class PerformanceCounter {
    public:
        PerformanceCounter(const char *name);
        ~PerformanceCounter();
        void doOperation(cl::Event &event, uint64_t flops, uint64_t bytes);

    private:
        static void eventCompleteCallBack(cl_event, cl_int, void *counter);
        static void report(const char *name, double runtime, uint64_t flops, uint64_t bytes);

    public:
       const char *name;
       uint64_t total_flops;
       uint64_t total_bytes;
       double   total_runtime;

   private:    
       std::function<void (cl_event)> callback;
};
