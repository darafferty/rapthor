#ifndef IDG_POWER_SENSOR_H_
#define IDG_POWER_SENSOR_H_

#if defined(HAVE_POWERSENSOR)
#include "powersensor.h"
#else
#include <string>

#include <omp.h>

#include "idg-config.h"
#endif


namespace powersensor {

    static std::string name_likwid("likwid");
    static std::string name_rapl("rapl");

    bool use_powersensor(const std::string name);

    #if not defined(HAVE_POWERSENSOR)
    class State {
        public:
            // Timestamp
            double timeAtRead;
    };
 
    class PowerSensor {

        public:
            virtual ~PowerSensor() {}
    
           virtual State read() = 0;
    
            virtual double seconds(const State &firstState, const State &secondState) = 0;
            virtual double Joules(const State &firstState, const State &secondState) = 0;
            virtual double Watt(const State &firstState, const State &secondState) = 0;
    };

    class DummyPowerSensor : public PowerSensor {
        public:
            static DummyPowerSensor* create();
    };

    #endif // end if not defined(HAVE_POWERSENSOR)

} // end namespace powersensor

#endif
