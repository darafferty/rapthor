#include <cstdio>

#include <CL/cl.hpp>

#include "idg-powersensor.h"

namespace idg {
    namespace kernel {
        namespace opencl {

            class PowerRecord {
                public:
                    void enqueue(cl::CommandQueue &queue);
                    static void getPower(cl_event, cl_int, void *userData);
                    PowerSensor *sensor;
                    PowerSensor::State state;
                    cl::Event event;
            };

        } // end namespace cuda
    } // end namespace kernel
} // end namespace idg

