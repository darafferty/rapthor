#include "idg-cuda.h"

#include "../common/common.h"

int main(int argc, char **argv) {
    run<idg::proxy::cuda::Jetson>();
}
