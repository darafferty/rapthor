#ifndef IDG_POWER_SENSOR_H_
#define IDG_POWER_SENSOR_H_

#include <string>
#include <omp.h>

#include "idg-config.h"

#if defined(HAVE_POWERSENSOR)
#include "powersensor.h"
#endif

namespace powersensor {

    static std::string name_likwid("likwid");
    static std::string name_rapl("rapl");
    static std::string name_nvml("nvml");
    static std::string name_arduino("tty");

    static std::string sensor_default("POWER_SENSOR");
    static std::string sensor_host("HOST_SENSOR");
    static std::string sensor_device("DEVICE_SENSOR");

    static const char *sensor_delimiter = ",";

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

    PowerSensor* get_power_sensor(
        const std::string name,
        const int i = 0);

} // end namespace powersensor

#endif
