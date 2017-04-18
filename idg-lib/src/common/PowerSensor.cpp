#include "PowerSensor.h"


namespace powersensor {

    bool use_powersensor(
        const std::string name,
        const char *power_sensor)
    {
        char *char_power_sensor = power_sensor ? (char *) power_sensor : getenv("POWER_SENSOR");
        if (char_power_sensor) {
            std::string str_power_sensor = std::string(char_power_sensor);
            return str_power_sensor.find(name) != std::string::npos;
        } else {
            return false;
        }
    }

    #if not defined(HAVE_POWERSENSOR)
    class DummyPowerSensor_ : public DummyPowerSensor {
        public:
            virtual State read();
            virtual double seconds(const State &firstState, const State &secondState) override;
            virtual double Joules(const State &firstState, const State &secondState) override;
            virtual double Watt(const State &firstState, const State &secondState) override;
    };
    
    DummyPowerSensor* DummyPowerSensor::create()
    {
        return new DummyPowerSensor_();
    }
    
    State DummyPowerSensor_::read() {
        State state;
        state.timeAtRead = omp_get_wtime();
        return state;
    }
    
    double DummyPowerSensor_::seconds(const State &firstState, const State &secondState) {
        return secondState.timeAtRead - firstState.timeAtRead;
    }
    
    double DummyPowerSensor_::Joules(const State &firstState, const State &secondState) {
        return Watt(firstState, secondState) * seconds(firstState, secondState);
    }
    
    
    double DummyPowerSensor_::Watt(const State &firstState, const State &secondState) {
        return 0;
    }

    #endif // end if not defined(HAVE_POWERSENSOR)

} // end namespace powersensor
