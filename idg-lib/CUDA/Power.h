#include "../PowerSensor/libPowerSensor.h"

#if defined MEASURE_POWER
PowerSensor *powerSensor;
#endif

struct Record
{
  public:
    void enqueue(cu::Stream&);
    mutable cu::Event event;

#if defined MEASURE_POWER
    PowerSensor::State state;

  private:
    static void getPower(CUstream, CUresult, void *userData);
#endif
};


void Record::enqueue(cu::Stream &stream) {
    stream.record(event);

#if defined MEASURE_POWER
    stream.addCallback(&Record::getPower, &state);
#endif
}

# if defined MEASURE_POWER
void Record::getPower(CUstream, CUresult, void *userData) {
  *(PowerSensor::State *) userData = (*powerSensor).read();
}
#endif
