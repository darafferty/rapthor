#include "idg-opencl.h"

#include "../../CPU/common/common.h"

int main(int argc, char **argv) {
    run<idg::proxy::opencl::Generic>();
}
