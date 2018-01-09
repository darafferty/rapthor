#include "Optimized.h"

#include "arch.h"

using namespace std;

namespace idg {
    namespace proxy {
        namespace cpu {

            // Constructor
            Optimized::Optimized(
                string libdir)
                : CPU(libdir)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
            }
        } // namespace cpu
    } // namespace proxy
} // namespace idg


#include "OptimizedC.h"
