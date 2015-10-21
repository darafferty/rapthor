#include "likwidPowerSensor.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <errno.h>
#include <omp.h>
#include <unistd.h>

#if !defined __MIC__
#include </usr/include/numa.h>
#endif

#include <likwid.h>


LikwidPowerSensor::LikwidPowerSensor(const char *dumpFileName)
:
  dumpFile(dumpFileName == 0 ? 0 : new std::ofstream(dumpFileName)),
  stop(false)
{
#if !defined __MIC__
  if (numa_init() != 0) {
    std::cerr << "numa_init() fails" << std::endl;
    exit(1);
  }

  if (topology_init() != 0) {
    std::cerr << "topology_init() fails" << std::endl;
    exit(1);
  }

  for (unsigned node = 0, nrNodes = numa_num_task_nodes(); node < nrNodes; node ++)
    power_init(node);
#endif

  if (dumpFile != 0) {
    if ((errno = pthread_mutex_init(&mutex, 0)) != 0) {
      perror("pthread_mutex_init");
      exit(1);
    }

    if ((errno = pthread_create(&thread, 0, &LikwidPowerSensor::IOthread, this)) != 0) {
      perror("pthread_create");
      exit(1);
    }
  }
}


LikwidPowerSensor::~LikwidPowerSensor()
{
  if (dumpFile != 0) {
    stop = true;

    if ((errno = pthread_join(thread, 0)) != 0)
      perror("pthread_join");

    if ((errno = pthread_mutex_destroy(&mutex)) != 0)
      perror("pthread_mutex_destroy");

    delete dumpFile;
  }
}


void LikwidPowerSensor::lock()
{
  if ((errno = pthread_mutex_lock(&mutex)) != 0) {
    perror("pthread_mutex_lock");
    exit(1);
  }
}


void LikwidPowerSensor::unlock()
{
  if ((errno = pthread_mutex_unlock(&mutex)) != 0) {
    perror("pthread_mutex_unlock");
    exit(1);
  }
}


void *LikwidPowerSensor::IOthread(void *arg)
{
  return static_cast<LikwidPowerSensor *>(arg)->IOthread();
}


void *LikwidPowerSensor::IOthread()
{
  State firstState = read(), currentState = firstState, previousState;

  while (!stop) {
    usleep(100);
    previousState = currentState;
    currentState  = read();

    lock();
    *dumpFile << "S " << seconds(firstState, currentState) << ' ' << Watt(previousState, currentState) << std::endl;
    unlock();
  }

  return 0;
}


void LikwidPowerSensor::mark(const State &state, const char *name, unsigned tag)
{
  if (dumpFile != 0) {
    lock();
    *dumpFile << "M " << state.timeAtRead << ' ' << tag << " \"" << (name == 0 ? "" : name) << '"' << std::endl;
    unlock();
  }
}


void LikwidPowerSensor::mark(const State &startState, const State &stopState, const char *name, unsigned tag)
{
  if (dumpFile != 0) {
    lock();
    *dumpFile << "M " << startState.timeAtRead << ' ' << stopState.timeAtRead << ' ' << tag << " \"" << (name == 0 ? "" : name) << '"' << std::endl;
    unlock();
  }
}


LikwidPowerSensor::State LikwidPowerSensor::read()
{
  State state;

#pragma omp critical (power)
  {
    state.timeAtRead = omp_get_wtime();
    state.consumedPKGenergy = 0;
    state.consumedDRAMenergy = 0;

#if !defined __MIC__
#define MSR_PKG_ENERGY_STATUS  0x611
#define MSR_DRAM_ENERGY_STATUS 0x619

    for (unsigned node = 0, nrNodes = numa_num_task_nodes(); node < nrNodes; node ++) {
      uint32_t energy;
      power_read(node, MSR_PKG_ENERGY_STATUS, &energy);
      state.consumedPKGenergy += energy;
      power_read(node, MSR_DRAM_ENERGY_STATUS, &energy);
      state.consumedDRAMenergy += energy;
    }
  }
#endif

  return state;
}


double LikwidPowerSensor::Joules(const State &firstState, const State &secondState)
{
  // multiply with energy unit in this function and not in read(), to be tolerant to counter overflows
  return (secondState.consumedPKGenergy  - firstState.consumedPKGenergy ) * power_info.domains[PKG].energyUnit +
	 (secondState.consumedDRAMenergy - firstState.consumedDRAMenergy) * power_info.domains[DRAM].energyUnit;
}


double LikwidPowerSensor::seconds(const State &firstState, const State &secondState)
{
  return secondState.timeAtRead - firstState.timeAtRead;
}


double LikwidPowerSensor::Watt(const State &firstState, const State &secondState)
{
  return Joules(firstState, secondState) / seconds(firstState, secondState);
}
