#ifndef IDG_POWER_SENSOR_H_
#define IDG_POWER_SENSOR_H_

#include <string>
#include <omp.h>

#include "idg-config.h"

#if defined(HAVE_POWERSENSOR)
#include "powersensor/PowerSensor.h"
#endif

namespace powersensor {

    static std::string name_likwid("likwid");
    static std::string name_rapl("rapl");
    static std::string name_nvml("nvml");
    static std::string name_arduino("tty");
    static std::string name_amdgpu("amdgpu");

    static std::string sensor_default("POWER_SENSOR");
    static std::string sensor_host("HOST_SENSOR");
    static std::string sensor_device("DEVICE_SENSOR");

    #if not defined(HAVE_POWERSENSOR)
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
    #endif

    PowerSensor* get_power_sensor(
        const std::string name,
        const unsigned i = 0);

    class DummyPowerSensor : public PowerSensor {
        public:
            DummyPowerSensor();
            virtual State read();
    };

} // end namespace powersensor

#endif
