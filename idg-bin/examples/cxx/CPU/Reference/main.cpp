#include "idg-cpu.h"

#include "common.h"


int main(int argc, char *argv[])
{
    run<idg::proxy::cpu::Reference>();

    return 0;
}
