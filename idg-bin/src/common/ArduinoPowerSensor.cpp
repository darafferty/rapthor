#include "ArduinoPowerSensor.h"

ArduinoPowerSensor::ArduinoPowerSensor(const char *device, const char *dumpFileName) {
    dumpFile = (dumpFileName == 0 ? 0 : new std::ofstream(dumpFileName));
    stop = false;
    lastMeasurement.microSeconds = 0;

    if ((fd = open(device, O_RDWR)) < 0) {
        perror("open device");
        exit(1);
    }

    // Configure port for 8N1 transmission
    struct termios options;

    // Get the current options for the port
    tcgetattr(fd, &options);

    // Sets the Input Baud Rate
    cfsetispeed(&options, B2000000);

    // Sets the Output Baud Rate
    cfsetospeed(&options, B2000000);

    // Set more options (TODO: add more comments)
    options.c_cflag = (options.c_cflag & ~CSIZE) | CS8;
    options.c_iflag = IGNBRK;
    options.c_lflag = 0;
    options.c_oflag = 0;
    options.c_cflag |= CLOCAL | CREAD;
    options.c_cc[VMIN] = sizeof(State::Measurement);
    options.c_cc[VTIME] = 0;
    options.c_iflag &= ~(IXON | IXOFF | IXANY);
    options.c_cflag &= ~(PARENB | PARODD);

    // Commit the options
    tcsetattr(fd, TCSANOW, &options);

    // Wait for the Arduino to reset
    sleep(2);

    // Flush anything already in the serial buffer
    tcflush(fd, TCIFLUSH);

    if ((errno = pthread_mutex_init(&mutex, 0)) != 0) {
        perror("pthread_mutex_init");
        exit(1);
    }

    // Initialize
    doMeasurement();

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
    State::Measurement currentState;

    ssize_t retval, bytesRead = 0;

    if (write(fd, "s", 1) != 1) {
        perror("write device");
        exit(1);
    }

    do {
        if ((retval = ::read(fd, (char *) &currentState.consumedEnergy + bytesRead, sizeof(State::Measurement) - bytesRead)) < 0) {
            perror("read device");
            exit(1);
        }
    } while ((bytesRead += retval) < sizeof(State::Measurement));

    if ((errno = pthread_mutex_lock(&mutex)) != 0) {
        perror("pthread_mutex_lock");
        exit(1);
    }

    if (lastMeasurement.microSeconds != currentState.microSeconds) {
        previousMeasurement = lastMeasurement;
        lastMeasurement = currentState;

        if (dumpFile != 0)
            *dumpFile << "S " << currentState.microSeconds / 1e6 << ' '
                              << (currentState.consumedEnergy - previousMeasurement.consumedEnergy) * (65536.0 / 511) /
                                 (currentState.microSeconds - previousMeasurement.microSeconds) << std::endl;
    }

    if ((errno = pthread_mutex_unlock(&mutex)) != 0) {
        perror("pthread_mutex_unlock");
        exit(1);
    }
}

PowerSensor::State ArduinoPowerSensor::read() {
    State state;

    if ((errno = pthread_mutex_lock(&mutex)) != 0) {
        perror("pthread_mutex_lock");
        exit(1);
    }

    state.previousMeasurement = previousMeasurement;
    state.lastMeasurement = lastMeasurement;
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
    uint32_t microSeconds = secondState.lastMeasurement.microSeconds - firstState.lastMeasurement.microSeconds;

    if (microSeconds != 0) {
        return (secondState.lastMeasurement.consumedEnergy -
                firstState.lastMeasurement.consumedEnergy) * (65536.0 / 511) / microSeconds;
    } else {
        return (secondState.lastMeasurement.consumedEnergy -
                secondState.previousMeasurement.consumedEnergy) * (65536.0 / 511) /
               (secondState.lastMeasurement.microSeconds - secondState.previousMeasurement.microSeconds);
    }
}
