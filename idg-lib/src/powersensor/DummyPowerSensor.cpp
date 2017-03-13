#include "DummyPowerSensor.h"

class DummyPowerSensor_ : public DummyPowerSensor {
    public:
        virtual PowerSensor::State read();
        virtual double seconds(const State &firstState, const State &secondState) override;
        virtual double Joules(const State &firstState, const State &secondState) override;
        virtual double Watt(const State &firstState, const State &secondState) override;
};

DummyPowerSensor* DummyPowerSensor::create()
{
    return new DummyPowerSensor_();
}

PowerSensor::State DummyPowerSensor_::read() {
    PowerSensor::State state;
    state.timeAtRead = omp_get_wtime();
    return state;
}

double DummyPowerSensor_::seconds(const State &firstState, const State &secondState) {
    return secondState.timeAtRead - firstState.timeAtRead;
}

double DummyPowerSensor_::Joules(const State &firstState, const State &secondState) {
    return Watt(firstState, secondState) * seconds(firstState, secondState);
}


double DummyPowerSensor_::Watt(const State &firstState, const State &secondState) {
    return 0;
}
