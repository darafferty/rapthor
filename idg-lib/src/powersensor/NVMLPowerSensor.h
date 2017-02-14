#ifndef IDG_NVML_POWER_SENSOR_H_
#define IDG_NVML_POWER_SENSOR_H_

#include "idg-config.h"

#include <fstream>
#ifdef BUILD_LIB_CUDA
#include "nvml.h"
#endif

#include "PowerSensor.h"

class NVMLPowerSensor : public PowerSensor {
    public:
        NVMLPowerSensor(const int device_number, const char *dumpFileName);
        ~NVMLPowerSensor();

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

#endif
