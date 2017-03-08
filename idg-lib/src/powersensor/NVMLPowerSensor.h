#ifndef IDG_NVML_POWER_SENSOR_H_
#define IDG_NVML_POWER_SENSOR_H_

#include "idg-config.h"

#include <fstream>

#include "PowerSensor.h"

class NVMLPowerSensor : public PowerSensor {
    public:
        static NVMLPowerSensor* create(const int device_number, const char *dumpFileName);
};

#endif
