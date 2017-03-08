#ifndef RAPL_POWER_SENSOR_H
#define RAPL_POWER_SENSOR_H

#include <fstream>
#include <iostream>

#include <unistd.h>

#include "PowerSensor.h"

class RaplPowerSensor : public PowerSensor {
    public:
        static RaplPowerSensor* create(const char *dumpFileName = 0);
};

#endif
