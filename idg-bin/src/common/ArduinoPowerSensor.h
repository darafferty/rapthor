#ifndef IDG_ARDUINO_POWER_SENSOR_H_
#define IDG_ARDUINO_POWER_SENSOR_H_

#include <iostream>
#include <fstream>

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include "PowerSensor.h"


class ArduinoPowerSensor : public PowerSensor {
    public:
        ArduinoPowerSensor(const char *device, const char *dumpFileName);

        State read();
        void mark(const State &, const char *name = 0, unsigned tag = 0);
        void mark(const State &start, const State &stop, const char *name = 0, unsigned tag = 0);

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
        void            doMeasurement();

        // State
        State::Measurement lastMeasurement;
        State::Measurement previousMeasurement;

        // Dump
        int fd;
        std::ofstream *dumpFile;
};

#endif
