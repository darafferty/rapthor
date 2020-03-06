#include "idg-cuda.h"

using namespace std;

using ProxyType = idg::proxy::cuda::Unified;

#include "../common/common.h"

int main(int argc, char *argv[])
{
    return compare_to_reference();
}
