#ifndef RAPL_POWER_SENSOR_H
#define RAPL_POWER_SENSOR_H

#include <fstream>
#include <iostream>

#include <unistd.h>

#include "PowerSensor.h"

class RaplPowerSensor : public PowerSensor {
    public:
        RaplPowerSensor(const char *dumpFileName = 0);
        ~RaplPowerSensor();

    State read();
    void mark(const State &, const char *name = 0, unsigned tag = 0);
    void mark(const State &start, const State &stop, const char *name = 0, unsigned tag = 0);

    virtual double Joules(const State &firstState, const State &secondState) override;
    virtual double seconds(const State &firstState, const State &secondState) override;
    virtual double Watt(const State &firstState, const State &secondState) override;

    private:
        // Thread
        pthread_t	    thread;
        pthread_mutex_t mutex;
        volatile bool   stop;
        static void     *IOthread(void *);
        void	        *IOthread();
        void	        lock();
        void            unlock();

        // Dump
        int             fd;
        std::ofstream   *dumpFile;
};

#endif
