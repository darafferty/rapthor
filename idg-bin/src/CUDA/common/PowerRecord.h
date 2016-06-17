#ifndef IDG_POWER_RECORD_H_
#define IDG_POWER_RECORD_H_

#include <cstdio>

#include "common/PowerSensor.h"
#include "CU.h"

class PowerRecord {
    public:
        void enqueue(cu::Stream &stream);
        static void getPower(CUstream, CUresult, void *userData);
        PowerSensor::State state;
        cu::Event event;
};

void init_powersensor();

#endif
