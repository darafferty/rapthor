#include "RaplPowerSensor.h"
#include "rapl-read.h"

class RaplPowerSensor_ : public RaplPowerSensor {
    public:
        RaplPowerSensor_(const char *dumpFileName);
        ~RaplPowerSensor_();

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

RaplPowerSensor* RaplPowerSensor::create(
    const char *dumpFileName)
{
    return new RaplPowerSensor_(dumpFileName);
}


RaplPowerSensor_::RaplPowerSensor_(const char *dumpFileName) :
    dumpFile(dumpFileName == 0 ? 0 : new std::ofstream(dumpFileName)),
    stop(false)
{
    if (dumpFile != 0) {
        if ((errno = pthread_mutex_init(&mutex, 0)) != 0) {
            perror("pthread_mutex_init");
            exit(1);
        }

        if ((errno = pthread_create(&thread, 0, &RaplPowerSensor_::IOthread, this)) != 0) {
            perror("pthread_create");
            exit(1);
        }
    }

    // Initialize rapl
    init_rapl();
}

RaplPowerSensor_::~RaplPowerSensor_() {
    if (dumpFile != 0) {
        stop = true;

        if ((errno = pthread_join(thread, 0)) != 0) {
            perror("pthread_join");
        }

        if ((errno = pthread_mutex_destroy(&mutex)) != 0) {
            perror("pthread_mutex_destroy");
        }

        delete dumpFile;
    }
}

void RaplPowerSensor_::lock() {
    if ((errno = pthread_mutex_lock(&mutex)) != 0) {
        perror("pthread_mutex_lock");
        exit(1);
    }
}


void RaplPowerSensor_::unlock() {
    if ((errno = pthread_mutex_unlock(&mutex)) != 0) {
        perror("pthread_mutex_unlock");
        exit(1);
    }
}

void *RaplPowerSensor_::IOthread(void *arg) {
    return static_cast<RaplPowerSensor_ *>(arg)->IOthread();
}


void *RaplPowerSensor_::IOthread() {
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

void RaplPowerSensor_::mark(const State &state, const char *name, unsigned tag) {
    if (dumpFile != 0) {
        lock();
        *dumpFile << "M " << state.timeAtRead << ' ' << tag << " \"" << (name == 0 ? "" : name) << '"' << std::endl;
        unlock();
    }
}

void RaplPowerSensor_::mark(const State &startState, const State &stopState, const char *name, unsigned tag) {
    if (dumpFile != 0) {
        lock();
        *dumpFile << "M " << startState.timeAtRead << ' ' << stopState.timeAtRead << ' ' << tag << " \"" << (name == 0 ? "" : name) << '"' << std::endl;
        unlock();
    }
}

RaplPowerSensor_::State RaplPowerSensor_::read() {
    State state;

    #pragma omp critical (power)
    {
        state = rapl_sysfs();
    }

    return state;
}

double RaplPowerSensor_::Joules(const State &firstState, const State &secondState) {
    return ((secondState.consumedEnergyPKG  - firstState.consumedEnergyPKG)) +
            (secondState.consumedEnergyDRAM - firstState.consumedEnergyDRAM);
}


double RaplPowerSensor_::seconds(const State &firstState, const State &secondState) {
    return secondState.timeAtRead - firstState.timeAtRead;
}


double RaplPowerSensor_::Watt(const State &firstState, const State &secondState) {
    return Joules(firstState, secondState) / seconds(firstState, secondState);
}
