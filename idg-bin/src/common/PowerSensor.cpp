#include "PowerSensor.h"

void PowerSensor::init(const char *device, const char *dumpFileName) {
    dumpFile = (dumpFileName == 0 ? 0 : new std::ofstream(dumpFileName));
    stop = false; 
    lastState.microSeconds = 0;

    #if defined(MEASURE_POWER_ARDUINO)
    if ((fd = open(device, O_RDWR)) < 0) {
        perror("open device");
        exit(1);
    }

    //Configure port for 8N1 transmission
    struct termios options;

    tcgetattr(fd, &options);		    //Gets the current options for the port
    cfsetispeed(&options, B2000000);	//Sets the Input Baud Rate
    cfsetospeed(&options, B2000000);	//Sets the Output Baud Rate

    options.c_cflag = (options.c_cflag & ~CSIZE) | CS8;
    options.c_iflag = IGNBRK;
    options.c_lflag = 0;
    options.c_oflag = 0;
    options.c_cflag |= CLOCAL | CREAD;
    options.c_cc[VMIN] = sizeof(State::MC_State);
    options.c_cc[VTIME] = 0;
    options.c_iflag &= ~(IXON | IXOFF | IXANY);
    options.c_cflag &= ~(PARENB | PARODD);

    /* commit the options */
    tcsetattr(fd, TCSANOW, &options);

    /* Wait for the Arduino to reset */
    sleep(2);
    /* Flush anything already in the serial buffer */
    tcflush(fd, TCIFLUSH);
    #endif

    if ((errno = pthread_mutex_init(&mutex, 0)) != 0) {
        perror("pthread_mutex_init");
        exit(1);
    }

    doMeasurement(); // initialise

    if ((errno = pthread_create(&thread, 0, &PowerSensor::IOthread, this)) != 0) {
        perror("pthread_create");
        exit(1);
    }
}


PowerSensor::~PowerSensor() {
    stop = true;
}


void *PowerSensor::IOthread(void *arg) {
    return static_cast<PowerSensor *>(arg)->IOthread();
}

void *PowerSensor::IOthread() {
    while (!stop)
        doMeasurement();
    void *retval;
    pthread_exit(retval);
    return retval;
}


void PowerSensor::doMeasurement() {
    State::MC_State currentState;

    #if defined(MEASURE_POWER_ARDUINO)
    ssize_t retval, bytesRead = 0;

    if (write(fd, "s", 1) != 1) {
        perror("write device");
        exit(1);
    }

    do {
        if ((retval = ::read(fd, (char *) &currentState.consumedEnergy + bytesRead, sizeof(State::MC_State) - bytesRead)) < 0) {
            perror("read device");
            exit(1);
        }
    } while ((bytesRead += retval) < sizeof(State::MC_State));
    #endif

    if ((errno = pthread_mutex_lock(&mutex)) != 0) {
        perror("pthread_mutex_lock");
        exit(1);
    }

    if (lastState.microSeconds != currentState.microSeconds) {
        previousState = lastState;
        lastState = currentState;

        #if defined(MEASURE_POWER_ARDUINO)
        if (dumpFile != 0)
            *dumpFile << "S " << currentState.microSeconds / 1e6 << ' '
                              << (currentState.consumedEnergy - previousState.consumedEnergy) * (65536.0 / 511) /
                                 (currentState.microSeconds - previousState.microSeconds) << std::endl;
        #endif
    }

    if ((errno = pthread_mutex_unlock(&mutex)) != 0) {
        perror("pthread_mutex_unlock");
        exit(1);
    }
}


void PowerSensor::mark(const State &state, const char *name, unsigned tag) {
    #if defined(MEASURE_POWER_ARDUINO)
    if (dumpFile != 0) {
        if ((errno = pthread_mutex_lock(&mutex)) != 0) {
            perror("pthread_mutex_lock");
            exit(1);
        }

        *dumpFile << "M " << state.lastState.microSeconds * 1e-6 << ' '
                  << tag << " \"" << (name == 0 ? "" : name) << '"' << std::endl;

        if ((errno = pthread_mutex_unlock(&mutex)) != 0) {
            perror("pthread_mutex_unlock");
            exit(1);
        }
    }
    #endif
}


void PowerSensor::mark(const State &startState, const State &stopState, const char *name, unsigned tag) {
    #if defined(MEASURE_POWER_ARDUINO)
    if (dumpFile != 0) {
        if ((errno = pthread_mutex_lock(&mutex)) != 0) {
            perror("pthread_mutex_lock");
            exit(1);
        }

        *dumpFile << "M " << startState.lastState.microSeconds * 1e-6 << ' '
                  << stopState.lastState.microSeconds * 1e-6 << ' ' << tag
                  << " \"" << (name == 0 ? "" : name) << '"' << std::endl;

        if ((errno = pthread_mutex_unlock(&mutex)) != 0) {
            perror("pthread_mutex_unlock");
            exit(1);
        }
    }
    #endif
}


PowerSensor::State PowerSensor::read() {
    State state;

    #if defined(MEASURE_POWER_ARDUINO)
    if ((errno = pthread_mutex_lock(&mutex)) != 0) {
        perror("pthread_mutex_lock");
        exit(1);
    }
    #endif

    state.previousState = previousState;
    state.lastState = lastState;
    state.timeAtRead = omp_get_wtime();

    #if defined(MEASURE_POWER_ARDUINO)
    if ((errno = pthread_mutex_unlock(&mutex)) != 0) {
        perror("pthread_mutex_unlock");
        exit(1);
    }
    #endif

    return state;
}


double PowerSensor::Joules(const State &firstState, const State &secondState) {
    return Watt(firstState, secondState) * seconds(firstState, secondState);
}


double PowerSensor::seconds(const State &firstState, const State &secondState) {
    return secondState.timeAtRead - firstState.timeAtRead;
}

double PowerSensor::Watt(const State &firstState, const State &secondState) {
    #if defined(MEASURE_POWER_ARDUINO)
    uint32_t microSeconds = secondState.lastState.microSeconds - firstState.lastState.microSeconds;

    if (microSeconds != 0)
        return (secondState.lastState.consumedEnergy -
                firstState.lastState.consumedEnergy) * (65536.0 / 511) / microSeconds;
    else // very short time
        return (secondState.lastState.consumedEnergy -
                secondState.previousState.consumedEnergy) * (65536.0 / 511) /
               (secondState.lastState.microSeconds - secondState.previousState.microSeconds);
    #else
    return 0;
    #endif
}
