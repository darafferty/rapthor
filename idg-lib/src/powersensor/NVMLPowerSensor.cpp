#include "NVMLPowerSensor.h"

#include <iostream>
#include <stdexcept>
#include <sstream>

#include <unistd.h>

#ifdef BUILD_LIB_CUDA
#include "nvml.h"

#define checkNVMLCall(val)  __checkNVMLCall((val), #val, __FILE__, __LINE__)

inline void __checkNVMLCall(
    nvmlReturn_t result,
    const char *const func,
    const char *const file,
    int const line)
{
    if (result != NVML_SUCCESS) {
        std::stringstream error;
        error << "NVML Error at " << file;
        error << ":" << line;
        error << " in function " << func;
        error << ": " << nvmlErrorString(result);
        error << std::endl;
        throw std::runtime_error(error.str());
    }
}
#endif


class NVMLPowerSensor_ : public NVMLPowerSensor {
    public:
        NVMLPowerSensor_(const int device_number, const char *dumpFileName);
        virtual ~NVMLPowerSensor_();

        virtual PowerSensor::State read();
        virtual double seconds(const State &firstState, const State &secondState) override;
        virtual double Joules(const State &firstState, const State &secondState) override;
        virtual double Watt(const State &firstState, const State &secondState) override;

    private:
        // Thread
        pthread_t       thread;
        pthread_mutex_t mutex;
        volatile bool   stop;
        static void     *IOthread(void *);
        void	        *IOthread();
        void	        lock();
        void            unlock();

        // State
        State previousState;

        // Dump
        int fd;
        std::ofstream *dumpFile;

#ifdef BUILD_LIB_CUDA
        nvmlDevice_t device;
#endif
};

NVMLPowerSensor* NVMLPowerSensor::create(
    const int device_number,
    const char *dumpFileName)
{
    return new NVMLPowerSensor_(device_number, dumpFileName);
}

NVMLPowerSensor_::NVMLPowerSensor_(
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

    if ((errno = pthread_create(&thread, 0, &NVMLPowerSensor_::IOthread, this)) != 0) {
        perror("pthread_create");
        exit(1);
    }
}


NVMLPowerSensor_::~NVMLPowerSensor_() {
    stop = true;
    if ((errno = pthread_join(thread, 0)) != 0) {
        perror("pthread_join");
    }

    if ((errno = pthread_mutex_destroy(&mutex)) != 0) {
        perror("pthread_mutex_destroy");
    }

#ifdef BUILD_LIB_CUDA
    checkNVMLCall(nvmlShutdown());
#endif
}


void NVMLPowerSensor_::lock() {
    if ((errno = pthread_mutex_lock(&mutex)) != 0) {
        perror("pthread_mutex_lock");
        exit(1);
    }
}


void NVMLPowerSensor_::unlock() {
    if ((errno = pthread_mutex_unlock(&mutex)) != 0) {
        perror("pthread_mutex_unlock");
        exit(1);
    }
}


void *NVMLPowerSensor_::IOthread(void *arg) {
    return static_cast<NVMLPowerSensor_ *>(arg)->IOthread();
}


void *NVMLPowerSensor_::IOthread() {
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


PowerSensor::State NVMLPowerSensor_::read() {
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


double NVMLPowerSensor_::seconds(const State &firstState, const State &secondState) {
    return secondState.timeAtRead - firstState.timeAtRead;
}


double NVMLPowerSensor_::Joules(const State &firstState, const State &secondState) {
#ifdef BUILD_LIB_CUDA
    return (secondState.consumedEnergyDevice - firstState.consumedEnergyDevice) * 1e-3;
#else
    return 0;
#endif
}


double NVMLPowerSensor_::Watt(const State &firstState, const State &secondState) {
    return Joules(firstState, secondState) / seconds(firstState, secondState);
}
