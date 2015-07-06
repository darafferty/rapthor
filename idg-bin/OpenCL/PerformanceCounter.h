#include <iostream>

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
        void report(double runtime, uint64_t flops, uint64_t bytes);

    public:
       const char *name;
       uint64_t total_flops;
       uint64_t total_bytes;
       double   total_runtime;

   private:
       uint64_t _flops;
       uint64_t _bytes;
};
