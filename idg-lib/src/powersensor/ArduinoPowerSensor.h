#ifndef ARDUINO_POWER_SENSOR_H_
#define ARDUINO_POWER_SENSOR_H_

#include <iostream>
#include <fstream>
#include <stdexcept>

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include "PowerSensor.h"


class ArduinoPowerSensor : public PowerSensor {
    public:
        static ArduinoPowerSensor* create(const char *device, const char *dumpFileName);
};

#endif
