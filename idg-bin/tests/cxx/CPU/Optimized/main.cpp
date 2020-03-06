#include "idg-cpu.h"

using namespace std;

using ProxyType = idg::proxy::cpu::Optimized;

#include "../common/common.h"

int main(int argc, char *argv[])
{
    return compare_to_reference();
}
