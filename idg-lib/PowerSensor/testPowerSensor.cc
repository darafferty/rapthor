#include <iostream>
#include <iomanip>
#include "libPowerSensor.h"

#include <cstdlib>
#include <iostream>

#include <inttypes.h>
#include <unistd.h>

#define MAX_MICRO_SECONDS 4000000


int main(int argc, char **argv)
{
  if (argc > 2) {
    std::cerr << "usage: " << argv[0] << " [device]" << std::endl;
    exit(1);
  }

  const char *device = argc == 2 ? argv[1] : "/dev/ttyUSB0";
  PowerSensor powerSensor(device);

  PowerSensor::State states[2];
  states[0] = powerSensor.read();

  for (uint32_t micros = 100, i = 1; micros <= MAX_MICRO_SECONDS; micros *= 2, i ^= 1) {
    usleep(micros);
    states[i] = powerSensor.read();

    std::cout << "exp. time: " << micros * 1e-6 << " s, "
      "measured: " << PowerSensor::seconds(states[i ^ 1], states[i]) << " s, " <<
      PowerSensor::Joules(states[i ^ 1], states[i]) << " J, " <<
      PowerSensor::Watt(states[i ^ 1], states[i]) << " W" <<
      std::endl;
  }
  
  return 0;
}
