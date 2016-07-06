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
        class State {
            public:
                double timeAtRead;
        };

        virtual State read() = 0;

        virtual double seconds(const State &firstState, const State &secondState) = 0;
        virtual double Joules(const State &firstState, const State &secondState) = 0;
        virtual double Watt(const State &firstState, const State &secondState) = 0;
};

#endif
