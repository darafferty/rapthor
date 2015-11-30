#include <cstdio> // remove()
#include <cstdlib>  // rand()
#include <ctime> // time() to init srand()
#include <complex>
#include <sstream>
#include <memory>
#include <dlfcn.h> // dlsym()
#include <omp.h> // omp_get_wtime
#include <libgen.h> // dirname() and basename()

#include "idg-config.h"
#include "MaxwellHaswellEP.h"
#if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
#include "auxiliary.h"
#endif

using namespace std;

namespace idg {
    namespace proxy {
        namespace hybrid {

        /// Constructors
        MaxwellHaswellEP::MaxwellHaswellEP(
            Parameters params)
        {
            #if defined(DEBUG)
            cout << "Maxwell-HaswellEP::" << __func__ << endl;
            cout << params;
            #endif
        }

        } // namespace hybrid
    } // namespace proxy
} // namespace idg
