#include "RaplPowerSensor.h"
#include "rapl-read.h"

RaplPowerSensor::RaplPowerSensor(const char *dumpFileName) :
    dumpFile(dumpFileName == 0 ? 0 : new std::ofstream(dumpFileName)),
    stop(false)
{
    if (dumpFile != 0) {
        if ((errno = pthread_mutex_init(&mutex, 0)) != 0) {
            perror("pthread_mutex_init");
            exit(1);
        }

        if ((errno = pthread_create(&thread, 0, &RaplPowerSensor::IOthread, this)) != 0) {
            perror("pthread_create");
            exit(1);
        }
    }

    // Initialize rapl
    init_rapl();
}

RaplPowerSensor::~RaplPowerSensor() {
    if (dumpFile != 0) {
        stop = true;

        if ((errno = pthread_join(thread, 0)) != 0) {
            perror("pthread_join");
        }

        if ((errno = pthread_mutex_destroy(&mutex)) != 0) {
            perror("pthread_mutex_destroy");
        }

        delete dumpFile;
    }
}

void RaplPowerSensor::lock() {
    if ((errno = pthread_mutex_lock(&mutex)) != 0) {
        perror("pthread_mutex_lock");
        exit(1);
    }
}


void RaplPowerSensor::unlock() {
    if ((errno = pthread_mutex_unlock(&mutex)) != 0) {
        perror("pthread_mutex_unlock");
        exit(1);
    }
}

void *RaplPowerSensor::IOthread(void *arg) {
    return static_cast<RaplPowerSensor *>(arg)->IOthread();
}


void *RaplPowerSensor::IOthread() {
    State firstState = read(), currentState = firstState, previousState;

    while (!stop) {
        usleep(100);
        previousState = currentState;
        currentState  = read();

        lock();
        if (dumpFile != 0) {
            *dumpFile << "S " << seconds(firstState, currentState) << ' ' << Watt(previousState, currentState) << std::endl;
        }
        unlock();
    }
}

void RaplPowerSensor::mark(const State &state, const char *name, unsigned tag) {
    if (dumpFile != 0) {
        lock();
        *dumpFile << "M " << state.timeAtRead << ' ' << tag << " \"" << (name == 0 ? "" : name) << '"' << std::endl;
        unlock();
    }
}

void RaplPowerSensor::mark(const State &startState, const State &stopState, const char *name, unsigned tag) {
    if (dumpFile != 0) {
        lock();
        *dumpFile << "M " << startState.timeAtRead << ' ' << stopState.timeAtRead << ' ' << tag << " \"" << (name == 0 ? "" : name) << '"' << std::endl;
        unlock();
    }
}

RaplPowerSensor::State RaplPowerSensor::read() {
    State state;

    #pragma omp critical (power)
    {
        state = rapl_sysfs();
    }

    return state;
}

double RaplPowerSensor::Joules(const State &firstState, const State &secondState) {
    return ((secondState.consumedEnergyPKG  - firstState.consumedEnergyPKG)) +
            (secondState.consumedEnergyDRAM - firstState.consumedEnergyDRAM);
}


double RaplPowerSensor::seconds(const State &firstState, const State &secondState) {
    return secondState.timeAtRead - firstState.timeAtRead;
}


double RaplPowerSensor::Watt(const State &firstState, const State &secondState) {
    return Joules(firstState, secondState) / seconds(firstState, secondState);
}
