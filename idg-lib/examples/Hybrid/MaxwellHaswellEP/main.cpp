#include "Hybrid/CUDA/MaxwellHaswellEP/idg.h"

#include "common.h"

int main(int argc, char **argv) {
    run<idg::proxy::hybrid::MaxwellHaswellEP>();
}
