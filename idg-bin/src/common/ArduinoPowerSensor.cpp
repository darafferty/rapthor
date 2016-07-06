#include "ArduinoPowerSensor.h"

ArduinoPowerSensor::ArduinoPowerSensor(const char *device, const char *dumpFileName) {
    dumpFile = (dumpFileName == 0 ? 0 : new std::ofstream(dumpFileName));
    stop = false;
    lastState.microSeconds = 0;

    if ((fd = open(device, O_RDWR)) < 0) {
        perror("open device");
        exit(1);
    }

    //Configure port for 8N1 transmission
    struct termios options;

    tcgetattr(fd, &options);		    // Gets the current options for the port
    cfsetispeed(&options, B2000000);	// Sets the Input Baud Rate
    cfsetospeed(&options, B2000000);	// Sets the Output Baud Rate

    options.c_cflag = (options.c_cflag & ~CSIZE) | CS8;
    options.c_iflag = IGNBRK;
    options.c_lflag = 0;
    options.c_oflag = 0;
    options.c_cflag |= CLOCAL | CREAD;
    options.c_cc[VMIN] = sizeof(ArduinoState::MC_State);
    options.c_cc[VTIME] = 0;
    options.c_iflag &= ~(IXON | IXOFF | IXANY);
    options.c_cflag &= ~(PARENB | PARODD);

    /* Commit the options */
    tcsetattr(fd, TCSANOW, &options);

    /* Wait for the Arduino to reset */
    sleep(2);

    /* Flush anything already in the serial buffer */
    tcflush(fd, TCIFLUSH);

    if ((errno = pthread_mutex_init(&mutex, 0)) != 0) {
        perror("pthread_mutex_init");
        exit(1);
    }

    doMeasurement(); // initialise

    if ((errno = pthread_create(&thread, 0, &ArduinoPowerSensor::IOthread, this)) != 0) {
        perror("pthread_create");
        exit(1);
    }
}

void *ArduinoPowerSensor::IOthread(void *arg) {
    return static_cast<ArduinoPowerSensor *>(arg)->IOthread();
}

void *ArduinoPowerSensor::IOthread() {
    while (!stop)
        doMeasurement();
    void *retval;
    pthread_exit(retval);
    return retval;
}


void ArduinoPowerSensor::doMeasurement() {
    ArduinoState::MC_State currentState;

    ssize_t retval, bytesRead = 0;

    if (write(fd, "s", 1) != 1) {
        perror("write device");
        exit(1);
    }

    do {
        if ((retval = ::read(fd, (char *) &currentState.consumedEnergy + bytesRead, sizeof(ArduinoState::MC_State) - bytesRead)) < 0) {
            perror("read device");
            exit(1);
        }
    } while ((bytesRead += retval) < sizeof(ArduinoState::MC_State));

    if ((errno = pthread_mutex_lock(&mutex)) != 0) {
        perror("pthread_mutex_lock");
        exit(1);
    }

    if (lastState.microSeconds != currentState.microSeconds) {
        previousState = lastState;
        lastState = currentState;

        if (dumpFile != 0)
            *dumpFile << "S " << currentState.microSeconds / 1e6 << ' '
                              << (currentState.consumedEnergy - previousState.consumedEnergy) * (65536.0 / 511) /
                                 (currentState.microSeconds - previousState.microSeconds) << std::endl;
    }

    if ((errno = pthread_mutex_unlock(&mutex)) != 0) {
        perror("pthread_mutex_unlock");
        exit(1);
    }
}

PowerSensor::State ArduinoPowerSensor::read() {
    ArduinoState state;

    if ((errno = pthread_mutex_lock(&mutex)) != 0) {
        perror("pthread_mutex_lock");
        exit(1);
    }

    state.previousState = previousState;
    state.lastState = lastState;
    state.timeAtRead = omp_get_wtime();

    if ((errno = pthread_mutex_unlock(&mutex)) != 0) {
        perror("pthread_mutex_unlock");
        exit(1);
    }

    return state;
}

double ArduinoPowerSensor::seconds(const State &firstState, const State &secondState) {
    return secondState.timeAtRead - firstState.timeAtRead;
}

double ArduinoPowerSensor::Joules(const State &firstState, const State &secondState) {
    return Watt(firstState, secondState) * seconds(firstState, secondState);
}

double ArduinoPowerSensor::Watt(const State &firstState, const State &secondState) {
    const ArduinoState *firstState_ = (ArduinoState *) &firstState;
    const ArduinoState *secondState_ = (ArduinoState *) &secondState;

    uint32_t microSeconds = secondState_->lastState.microSeconds - firstState_->lastState.microSeconds;

    if (microSeconds != 0) {
        return (secondState_->lastState.consumedEnergy -
                firstState_->lastState.consumedEnergy) * (65536.0 / 511) / microSeconds;
    } else {
        return (secondState_->lastState.consumedEnergy -
                secondState_->previousState.consumedEnergy) * (65536.0 / 511) /
               (secondState_->lastState.microSeconds - secondState_->previousState.microSeconds);
    }
}
