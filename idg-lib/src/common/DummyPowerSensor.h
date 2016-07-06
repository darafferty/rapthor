#ifndef IDG_DUMMY_POWER_SENSOR_H_
#define IDG_DUMMY_POWER_SENSOR_H_

#include "PowerSensor.h"

class DummyPowerSensor : public PowerSensor {
    public:
        DummyPowerSensor();
        virtual PowerSensor::State read();
        virtual double seconds(const State &firstState, const State &secondState) override;
        virtual double Joules(const State &firstState, const State &secondState) override;
        virtual double Watt(const State &firstState, const State &secondState) override;
};

#endif
