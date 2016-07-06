#ifndef IDG_POWER_SENSOR_H_
#define IDG_POWER_SENSOR_H_

#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cerrno>

#include <pthread.h>
#include <omp.h>

#include "idg-config.h"

class PowerSensor {
    public:
        struct State {
            struct MC_State {
                int32_t  consumedEnergy = 0;
                uint32_t microSeconds   = 0;
            } previousState, lastState;
            double timeAtRead;
        };

        PowerSensor();
        ~PowerSensor();

        State read();

        static double Joules(const State &firstState, const State &secondState);
        static double seconds(const State &firstState, const State &secondState);
        static double Watt(const State &firstState, const State &secondState);

    private:
        pthread_t	  thread;
        pthread_mutex_t mutex;
        volatile bool stop;
        State::MC_State previousState, lastState;

        static void   *IOthread(void *);
        void	  *IOthread();
        void	  doMeasurement();
};

#endif
