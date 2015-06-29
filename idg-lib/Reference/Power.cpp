#include "Power.h"

PowerSensor::PowerSensor() {
    #if MEASURE_POWER && !defined __MIC__
    if (!lock_check()) {
        std::cerr << "Access to performance counters is locked" << std::endl;
        exit(1);
    }

    if (cpuid_init() == EXIT_FAILURE) {
        std::cerr << "CPU not supported" << std::endl;
        exit(1);
    }

    if (numa_init() != 0) {
        std::cerr << "numa_init() fails" << std::endl;
        exit(1);
    }

    accessClient_init(&powerSocket);
    msr_init(powerSocket);
    timer_init();
    power_init(0);
    #endif
}

PowerSensor::~PowerSensor() {
    #if MEASURE_POWER && !defined __MIC__
    msr_finalize();
    #endif
}

PowerSensor::State PowerSensor::read() {
    State state;

    #pragma omp critical(power)
    {
    state.seconds = omp_get_wtime();

    #if MEASURE_POWER && !defined __MIC__
    state.consumedEnergy = 0;
    unsigned nrNodes     = numa_num_task_nodes();

    for (unsigned node = 0; node < nrNodes; node ++) {
        state.consumedEnergy += power_read(node, power_regs[PKG]);
        state.consumedEnergy += power_read(node, power_regs[DRAM]) / 4;
    }
    #endif
    }

    return state;
}

double PowerSensor::Joules(const State &firstState, const State &secondState) {
    #if MEASURE_POWER && !defined __MIC__
    return (secondState.consumedEnergy - firstState.consumedEnergy) * power_info.energyUnit;
    #else
    return 0;
    #endif
}

double PowerSensor::seconds(const State &firstState, const State &secondState) {
    return secondState.seconds - firstState.seconds;
}

double PowerSensor::Watt(const State &firstState, const State &secondState) {
    return Joules(firstState, secondState) / seconds(firstState, secondState);
}

