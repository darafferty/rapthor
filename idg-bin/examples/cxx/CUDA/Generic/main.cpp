#include "idg-cuda.h"

#include "../../CPU/common/common.h"

int main(int argc, char **argv) {
    run<idg::proxy::cuda::Generic>();
}
