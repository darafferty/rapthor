#include "PowerRecord.h"

static PowerSensor powerSensor;

void PowerRecord::enqueue(cu::Stream &stream) {
    stream.record(event);
    stream.addCallback((CUstreamCallback) &PowerRecord::getPower, &state);
}

void PowerRecord::getPower(CUstream, CUresult, void *userData) {
    *static_cast<PowerSensor::State *>(userData) = powerSensor.read();
}

void init_powersensor() {
    #if defined(MEASURE_POWER_ARDUINO)
    const char *str_power_sensor = getenv("POWER_SENSOR");
    if (!str_power_sensor) str_power_sensor = POWER_SENSOR;
    const char *str_power_file = getenv("POWER_FILE");
    if (!str_power_file) str_power_file = POWER_FILE;
    std::cout << "Opening power sensor: " << str_power_sensor << std::endl;
    std::cout << "Writing power consumption to file: " << str_power_file << std::endl;
    powerSensor.init(str_power_sensor, str_power_file);
    #else
    powerSensor.init();
    #endif
}
