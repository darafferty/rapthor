#ifndef IDG_DUMMY_POWER_SENSOR_H_
#define IDG_DUMMY_POWER_SENSOR_H_

#include "PowerSensor.h"

class DummyPowerSensor : public PowerSensor {
    public:
        static DummyPowerSensor* create();
};

#endif
