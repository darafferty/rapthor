#include <cstdio>

#include <CL/cl.hpp>

#include "idg-common.h"

namespace idg {
    namespace kernel {
        namespace opencl {

            class PowerRecord {
                public:
                    PowerRecord();
                    PowerRecord(powersensor::PowerSensor *sensor);

                    void enqueue(cl::CommandQueue &queue);
                    static void getPower(cl_event, cl_int, void *userData);
                    powersensor::PowerSensor *sensor;
                    powersensor::State state;
                    cl::Event event;
            };

        } // end namespace cuda
    } // end namespace kernel
} // end namespace idg

