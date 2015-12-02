#include "CUDA/Maxwell/idg.h"

#include "common.h"

int main(int argc, char **argv) {
    run<idg::proxy::cuda::Maxwell>();
}
