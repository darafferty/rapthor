#include "idg-opencl.h"
#include "common.h"

using namespace std;

int main(int argc, char *argv[])
{
    // Compares to reference implementation
    int info = compare_to_reference<idg::proxy::opencl::Generic>();

    return info;
}
