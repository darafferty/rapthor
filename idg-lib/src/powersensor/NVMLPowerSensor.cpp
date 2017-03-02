#include "NVMLPowerSensor.h"

#include <iostream>
#include <stdexcept>

#include <unistd.h>

#ifdef BUILD_LIB_CUDA
#define checkNVMLCall(val)  __checkNVMLCall((val), #val, __FILE__, __LINE__)

inline void __checkNVMLCall(
    nvmlReturn_t result,
    const char *const func,
    const char *const file,
    int const line)
{
    if (result != NVML_SUCCESS) {
        std::cerr << "NVML Error at " << file;
        std::cerr << ":" << line;
        std::cerr << " in function " << func;
        std::cerr << ": " << nvmlErrorString(result);
        std::cerr << std::endl;
        exit(EXIT_FAILURE);
    }

}
#endif


NVMLPowerSensor::NVMLPowerSensor(
    const int device_number,
    const char *dumpFileName)
{
    dumpFile = (dumpFileName == 0 ? 0 : new std::ofstream(dumpFileName));
    stop = false;

#ifdef BUILD_LIB_CUDA
    checkNVMLCall(nvmlInit());
    checkNVMLCall(nvmlDeviceGetHandleByIndex(device_number, &device));
#endif

    previousState = read();
    previousState.consumedEnergyDevice = 0;

    if ((errno = pthread_mutex_init(&mutex, 0)) != 0) {
        perror("pthread_mutex_init");
        exit(1);
    }

    if ((errno = pthread_create(&thread, 0, &NVMLPowerSensor::IOthread, this)) != 0) {
        perror("pthread_create");
        exit(1);
    }
}


NVMLPowerSensor::~NVMLPowerSensor() {
#ifdef BUILD_LIB_CUDA
    checkNVMLCall(nvmlShutdown());
#endif
}


void NVMLPowerSensor::lock() {
    if ((errno = pthread_mutex_lock(&mutex)) != 0) {
        perror("pthread_mutex_lock");
        exit(1);
    }
}


void NVMLPowerSensor::unlock() {
    if ((errno = pthread_mutex_unlock(&mutex)) != 0) {
        perror("pthread_mutex_unlock");
        exit(1);
    }
}


void *NVMLPowerSensor::IOthread(void *arg) {
    return static_cast<NVMLPowerSensor *>(arg)->IOthread();
}


void *NVMLPowerSensor::IOthread() {
    State firstState = read(), currentState;

    while (!stop) {
        currentState = read();

        lock();
        if (dumpFile != 0) {
            *dumpFile << "S " << seconds(firstState, currentState) << ' ' << Watt(previousState, currentState) << std::endl;
        }
        unlock();

        previousState = currentState;
    }

    void *retval;
    pthread_exit(retval);
    return retval;
}


PowerSensor::State NVMLPowerSensor::read() {
    State state;
    state.timeAtRead = omp_get_wtime();

#ifdef BUILD_LIB_CUDA
    unsigned int power;
    checkNVMLCall(nvmlDeviceGetPowerUsage(device, &power));

    state.instantaneousPower = power;

    state.consumedEnergyDevice = previousState.consumedEnergyDevice;
    float averagePower = (state.instantaneousPower + previousState.instantaneousPower) / 2;
    float timeElapsed = (state.timeAtRead - previousState.timeAtRead);
    state.consumedEnergyDevice += averagePower * timeElapsed;
#endif

    return state;
}


double NVMLPowerSensor::seconds(const State &firstState, const State &secondState) {
    return secondState.timeAtRead - firstState.timeAtRead;
}


double NVMLPowerSensor::Joules(const State &firstState, const State &secondState) {
#ifdef BUILD_LIB_CUDA
    return (secondState.consumedEnergyDevice - firstState.consumedEnergyDevice) * 1e-3;
#else
    return 0;
#endif
}


double NVMLPowerSensor::Watt(const State &firstState, const State &secondState) {
    return Joules(firstState, secondState) / seconds(firstState, secondState);
}
