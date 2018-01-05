#include "Reference.h"

using namespace std;

namespace idg {
    namespace proxy {
        namespace cpu {

            // Constructor
            Reference::Reference(
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

#include "ReferenceC.h"
