#ifndef IDG_POWER_RECORD_H_
#define IDG_POWER_RECORD_H_

#include <cstdio>

#include "common/PowerSensor.h"
#include "CU.h"

namespace idg {
    namespace proxy {
        namespace cuda {

            class PowerRecord {
                public:
                    void enqueue(cu::Stream &stream);
                    static void getPower(CUstream, CUresult, void *userData);
                    PowerSensor *sensor;
                    PowerSensor::State state;
                    cu::Event event;
            };

        } // end namespace cuda
    } // end namespace proxy
} // end namespace idg

#endif
