#ifndef IDG_POWER_RECORD_H_
#define IDG_POWER_RECORD_H_

#include <cstdio>

#include "idg-common.h"

#include "CU.h"

namespace idg {
    namespace kernel {
        namespace cuda {

            class PowerRecord {
                public:
                    PowerRecord();
                    PowerRecord(powersensor::PowerSensor *sensor);

                    void enqueue(cu::Stream &stream);
                    static void getPower(CUstream, CUresult, void *userData);
                    powersensor::PowerSensor *sensor;
                    powersensor::State state;
                    cu::Event event;
            };

        } // end namespace cuda
    } // end namespace kernel
} // end namespace idg

#endif