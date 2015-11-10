#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include <errno.h>
#include <fcntl.h>
#include <omp.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <termios.h>
#include <unistd.h>
#include <inttypes.h>
#include <pthread.h>


#if defined(MEASURE_POWER_ARDUINO)
#define _QUOTE(str) #str
#define QUOTE(str) _QUOTE(str)
#define STR_POWER_SENSOR QUOTE(POWER_SENSOR)
#define STR_POWER_FILE QUOTE(POWER_FILE)
#endif


class PowerSensor
{
  public:
    struct State
    {
      struct MC_State
      {
	int32_t  consumedEnergy = 0;
	uint32_t microSeconds = 0;
      } previousState, lastState;

      double timeAtRead;
    };

    PowerSensor(const char *device = "/dev/ttyUSB0", const char *dumpFileName = 0);
    ~PowerSensor();

    State read();
    void mark(const State &, const char *name = 0, unsigned tag = 0);
    void mark(const State &start, const State &stop, const char *name = 0, unsigned tag = 0);

    static double Joules(const State &firstState, const State &secondState);
    static double seconds(const State &firstState, const State &secondState);
    static double Watt(const State &firstState, const State &secondState);

  private:
    int		  fd;
    std::ofstream *dumpFile;
    pthread_t	  thread;
    pthread_mutex_t mutex;
    volatile bool stop;
    State::MC_State previousState, lastState;

    static void   *IOthread(void *);
    void	  *IOthread();
    void	  doMeasurement();
};
