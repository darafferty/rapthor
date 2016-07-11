#include "DummyPowerSensor.h"

DummyPowerSensor::DummyPowerSensor() {};


PowerSensor::State DummyPowerSensor::read() {
    PowerSensor::State state;
    state.timeAtRead = omp_get_wtime();
    return state;
}

double DummyPowerSensor::seconds(const State &firstState, const State &secondState) {
    return secondState.timeAtRead - firstState.timeAtRead;
}

double DummyPowerSensor::Joules(const State &firstState, const State &secondState) {
    return Watt(firstState, secondState) * seconds(firstState, secondState);
}


double DummyPowerSensor::Watt(const State &firstState, const State &secondState) {
    return 0;
}
