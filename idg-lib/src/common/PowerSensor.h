#ifndef IDG_POWER_SENSOR_H_
#define IDG_POWER_SENSOR_H_

#include <string>
#include <omp.h>

#include "idg-config.h"

namespace powersensor {

    static std::string name_likwid("likwid");
    static std::string name_rapl("rapl");
    static std::string name_nvml("nvml");
    static std::string name_arduino("tty");
    static std::string name_amdgpu("amdgpu");

    static std::string sensor_default("POWER_SENSOR");
    static std::string sensor_host("HOST_SENSOR");
    static std::string sensor_device("DEVICE_SENSOR");

    class State {
        public:
            double timeAtRead;
            double joulesAtRead;
    };

    class PowerSensor {
        public:
            virtual ~PowerSensor() {}

            virtual State read() = 0;

            virtual double seconds(const State &firstState, const State &secondState) = 0;
            virtual double Joules(const State &firstState, const State &secondState) = 0;
            virtual double Watt(const State &firstState, const State &secondState) = 0;
    };

    PowerSensor* get_power_sensor(
        const std::string name,
        const unsigned i = 0);

    class DummyPowerSensor : public PowerSensor {
        public:
            static DummyPowerSensor* create();
    };

    namespace likwid {
        class LikwidPowerSensor : public PowerSensor {
            public:
                static LikwidPowerSensor* create();
        };
    }

    namespace rapl {
        class RaplPowerSensor : public PowerSensor {
            public:
                static RaplPowerSensor* create(const char*);
        };
    }

    namespace nvml {
        class NVMLPowerSensor : public PowerSensor {
            public:
                static NVMLPowerSensor* create(const int, const char*);
        };
    }

    namespace arduino {
        class ArduinoPowerSensor : public PowerSensor {
            public:
                static ArduinoPowerSensor* create(const char*, const char*);
        };
    }

    namespace amdgpu {
        class AMDGPUPowerSensor : public PowerSensor {
            public:
                static AMDGPUPowerSensor* create(const unsigned, const char*);
        };
    }
} // end namespace powersensor

#endif
