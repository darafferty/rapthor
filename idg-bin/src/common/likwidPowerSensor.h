#if !defined POWER_SENSOR_H
#define POWER_SENSOR_H

#include <inttypes.h>
#include <pthread.h>

#include <fstream>
#include <vector>


class PowerSensor
{
  public:
    struct State
    {
      unsigned consumedPKGenergy, consumedDRAMenergy;
      double timeAtRead;
    };

    PowerSensor(const char *dumpFileName = 0);
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

    void	  lock(), unlock();

    static void   *IOthread(void *);
    void	  *IOthread();
};

#endif
