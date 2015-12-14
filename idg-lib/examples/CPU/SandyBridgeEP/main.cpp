#include "idg-cpu.h"

#include "../common/common.h"


int main(int argc, char *argv[])
{
    run<idg::proxy::cpu::SandyBridgeEP>();

    return 0;
}
