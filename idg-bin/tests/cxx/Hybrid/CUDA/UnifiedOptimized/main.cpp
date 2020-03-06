#include "idg-hybrid-cuda.h"

using namespace std;

using ProxyType = idg::proxy::hybrid::UnifiedOptimized;

#include "../common/common.h"

int main(int argc, char *argv[])
{
    return compare_to_reference();
}
