#include "LikwidPowerSensor.h"

#if defined(HAVE_NUMA)
#if !defined __MIC__
#include </usr/include/numa.h>
#endif
#endif

#if defined(HAVE_LIKWID)
#include <likwid.h>
#endif

class LikwidPowerSensor_ : public LikwidPowerSensor {
    public:
        LikwidPowerSensor_(const char *dumpFileName);
        ~LikwidPowerSensor_();

    State read();
    void mark(const State &, const char *name = 0, unsigned tag = 0);
    void mark(const State &start, const State &stop, const char *name = 0, unsigned tag = 0);

    virtual double Joules(const State &firstState, const State &secondState) override;
    virtual double seconds(const State &firstState, const State &secondState) override;
    virtual double Watt(const State &firstState, const State &secondState) override;

    private:
        // Thread
        pthread_t	    thread;
        pthread_mutex_t mutex;
        volatile bool   stop;
        static void     *IOthread(void *);
        void	        *IOthread();
        void	        lock();
        void            unlock();

        // Dump
        int             fd;
        std::ofstream   *dumpFile;
};

LikwidPowerSensor* LikwidPowerSensor::create(
    const char *dumpFileName)
{
    return new LikwidPowerSensor_(dumpFileName);
}

LikwidPowerSensor_::LikwidPowerSensor_(const char *dumpFileName) :
    dumpFile(dumpFileName == 0 ? 0 : new std::ofstream(dumpFileName)),
    stop(false)
{
    #if defined(HAVE_NUMA)
    #if !defined __MIC__
    if (numa_init() != 0) {
        std::cerr << "numa_init() fails" << std::endl;
        exit(1);
    }

    if (topology_init() != 0) {
        std::cerr << "topology_init() fails" << std::endl;
        exit(1);
    }

    for (unsigned node = 0, nrNodes = numa_num_task_nodes(); node < nrNodes; node ++) {
        power_init(node);
    }
    #endif

    if (dumpFile != 0) {
        if ((errno = pthread_mutex_init(&mutex, 0)) != 0) {
            perror("pthread_mutex_init");
            exit(1);
        }

        if ((errno = pthread_create(&thread, 0, &LikwidPowerSensor_::IOthread, this)) != 0) {
            perror("pthread_create");
            exit(1);
        }
    }
    #endif
}


LikwidPowerSensor_::~LikwidPowerSensor_() {
    if (dumpFile != 0) {
        stop = true;

        #if defined(HAVE_LIKWID)
        if ((errno = pthread_join(thread, 0)) != 0) {
            perror("pthread_join");
        }

        if ((errno = pthread_mutex_destroy(&mutex)) != 0) {
            perror("pthread_mutex_destroy");
        }
        #endif

        delete dumpFile;
    }
}


void LikwidPowerSensor_::lock() {
    if ((errno = pthread_mutex_lock(&mutex)) != 0) {
        perror("pthread_mutex_lock");
        exit(1);
    }
}


void LikwidPowerSensor_::unlock() {
    if ((errno = pthread_mutex_unlock(&mutex)) != 0) {
        perror("pthread_mutex_unlock");
        exit(1);
    }
}


void *LikwidPowerSensor_::IOthread(void *arg) {
    return static_cast<LikwidPowerSensor_ *>(arg)->IOthread();
}


void *LikwidPowerSensor_::IOthread() {
    State firstState = read(), currentState = firstState, previousState;

    while (!stop) {
        usleep(100);
        previousState = currentState;
        currentState  = read();

        lock();
        if (dumpFile != 0) {
            *dumpFile << "S " << seconds(firstState, currentState) << ' ' << Watt(previousState, currentState) << std::endl;
        }
        unlock();
    }
}


void LikwidPowerSensor_::mark(const State &state, const char *name, unsigned tag) {
    if (dumpFile != 0) {
        lock();
        *dumpFile << "M " << state.timeAtRead << ' ' << tag << " \"" << (name == 0 ? "" : name) << '"' << std::endl;
        unlock();
    }
}


void LikwidPowerSensor_::mark(const State &startState, const State &stopState, const char *name, unsigned tag) {
    if (dumpFile != 0) {
        lock();
        *dumpFile << "M " << startState.timeAtRead << ' ' << stopState.timeAtRead << ' ' << tag << " \"" << (name == 0 ? "" : name) << '"' << std::endl;
        unlock();
    }
}


LikwidPowerSensor_::State LikwidPowerSensor_::read() {
    State state;

    state.timeAtRead = omp_get_wtime();
    state.consumedEnergyPKG = 0;
    state.consumedEnergyDRAM = 0;

    #if defined(HAVE_LIKWID)
    #pragma omp critical (power)
    {
        #if !defined __MIC__
        #define MSR_PKG_ENERGY_STATUS  0x611
        #define MSR_DRAM_ENERGY_STATUS 0x619

        for (unsigned node = 0, nrNodes = numa_num_task_nodes(); node < nrNodes; node ++) {
            uint32_t energy;
            power_read(node, MSR_PKG_ENERGY_STATUS, &energy);
            state.consumedEnergyPKG += energy;
            power_read(node, MSR_DRAM_ENERGY_STATUS, &energy);
            state.consumedEnergyDRAM += energy;
        }
    }
    #endif
    #endif

    return state;
}


double LikwidPowerSensor_::Joules(const State &firstState, const State &secondState) {
    #if defined(HAVE_LIKWID)
    // Multiply with energy unit in this function and not in read(), to be tolerant to counter overflows
    return (secondState.consumedEnergyPKG  - firstState.consumedEnergyPKG ) * power_info.domains[PKG].energyUnit +
           (secondState.consumedEnergyDRAM - firstState.consumedEnergyDRAM) * power_info.domains[DRAM].energyUnit;
    #else
    return 0;
    #endif
}


double LikwidPowerSensor_::seconds(const State &firstState, const State &secondState) {
    return secondState.timeAtRead - firstState.timeAtRead;
}


double LikwidPowerSensor_::Watt(const State &firstState, const State &secondState) {
    return Joules(firstState, secondState) / seconds(firstState, secondState);
}
