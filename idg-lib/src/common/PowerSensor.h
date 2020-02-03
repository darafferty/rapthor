#ifndef IDG_POWER_SENSOR_H_
#define IDG_POWER_SENSOR_H_

#include <string>

#include "idg-config.h"

namespace powersensor {

    static std::string sensor_default("POWER_SENSOR");
    static std::string sensor_host("HOST_SENSOR");
    static std::string sensor_device("DEVICE_SENSOR");

    #if not defined(POWERSENSOR_DEFINED)
    class State {
        public:
        double timeAtRead   = 0;
        double joulesAtRead = 0;
    };

    class PowerSensor {
        public:
            virtual ~PowerSensor();
            virtual State read() = 0;
            static double seconds(const State &firstState, const State &secondState);
            static double Joules(const State &firstState, const State &secondState);
            static double Watt(const State &firstState, const State &secondState);
    };

    PowerSensor* get_power_sensor(
        const std::string name,
        const unsigned i = 0);
    #endif

} // end namespace powersensor

#endif