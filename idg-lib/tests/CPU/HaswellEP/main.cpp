#include "../common/common.h"

using namespace std;

int main(int argc, char *argv[])
{
    // Reads the following from ENV:
    // NR_STATIONS, NR_CHANNELS, NR_TIMESTEPS, NR_TIMESLOTS, IMAGESIZE,
    // GRIDSIZE; if non-default jobsize wanted, set jobsizes parameters.

    int info = run_test<idg::proxy::cpu::HaswellEP>();

    return info;
}
