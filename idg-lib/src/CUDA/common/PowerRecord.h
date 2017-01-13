#ifndef IDG_POWER_RECORD_H_
#define IDG_POWER_RECORD_H_

#include <cstdio>

#include "idg-powersensor.h"

#include "CU.h"

namespace idg {
    namespace kernel {
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
    } // end namespace kernel
} // end namespace idg

#endif
