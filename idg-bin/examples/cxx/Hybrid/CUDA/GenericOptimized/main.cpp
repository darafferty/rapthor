#include "idg-hybrid-cuda.h"

#include "common.h"

int main(int argc, char **argv) {
    run<idg::proxy::hybrid::GenericOptimized>();
}
