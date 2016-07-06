#include "PowerSensor.h"

PowerSensor::PowerSensor() {};

PowerSensor::~PowerSensor() {
    stop = true;
}

void *PowerSensor::IOthread(void *arg) {
    return static_cast<PowerSensor *>(arg)->IOthread();
}

void *PowerSensor::IOthread() {
    while (!stop)
        doMeasurement();
    void *retval;
    pthread_exit(retval);
    return retval;
}


void PowerSensor::doMeasurement() {
    State::MC_State currentState;

    if ((errno = pthread_mutex_lock(&mutex)) != 0) {
        perror("pthread_mutex_lock");
        exit(1);
    }

    if (lastState.microSeconds != currentState.microSeconds) {
        previousState = lastState;
        lastState = currentState;
    }

    if ((errno = pthread_mutex_unlock(&mutex)) != 0) {
        perror("pthread_mutex_unlock");
        exit(1);
    }
}

PowerSensor::State PowerSensor::read() {
    State state;
    state.previousState = previousState;
    state.lastState = lastState;
    state.timeAtRead = omp_get_wtime();
    return state;
}

double PowerSensor::Joules(const State &firstState, const State &secondState) {
    return Watt(firstState, secondState) * seconds(firstState, secondState);
}

double PowerSensor::seconds(const State &firstState, const State &secondState) {
    return secondState.timeAtRead - firstState.timeAtRead;
}

double PowerSensor::Watt(const State &firstState, const State &secondState) {
    return 0;
}
