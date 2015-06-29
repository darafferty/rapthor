#if USE_LIKWID

#include <iostream>

#include <stdint.h>
#include <omp.h>

extern "C" {
#include <likwid/accessClient.h>
#include <likwid/cpuid.h>
#include <likwid/lock.h>
#include <likwid/msr.h>
#include <likwid/numa.h>
#include <likwid/power.h>
#include <likwid/timer.h>
}

#include </usr/include/numa.h>

class PowerSensor {
    public:
        class State {
            public:
            #if MEASURE_POWER && !defined __MIC__
                uint32_t consumedEnergy;
            #endif
                double seconds;
        };

        PowerSensor();
        ~PowerSensor();

        State read();

        static double  Joules(const State &firstState, const State &secondState);
        static double seconds(const State &firstState, const State &secondState);
        static double    Watt(const State &firstState, const State &secondState);

    private:
      int powerSocket;
};

static PowerSensor powerSensor;

#endif // USE_LIKWID
