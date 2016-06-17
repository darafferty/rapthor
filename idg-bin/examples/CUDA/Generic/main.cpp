#include "idg-cuda.h"

#include "../common/common-new.h"

int main(int argc, char **argv) {
    run<idg::proxy::cuda::Generic>();
}
