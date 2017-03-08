#ifndef LIKWID_POWER_SENSOR_H
#define LIKWID_POWER_SENSOR_H

#include <fstream>
#include <iostream>

#include <unistd.h>

#include "PowerSensor.h"

class LikwidPowerSensor : public PowerSensor {
    public:
        static LikwidPowerSensor* create(const char *dumpFileName = 0);
};

#endif
