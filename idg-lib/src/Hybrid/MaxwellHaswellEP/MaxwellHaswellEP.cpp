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
using namespace idg::kernel;

namespace idg {
    namespace proxy {
        namespace hybrid {

            /// Constructors
            MaxwellHaswellEP::MaxwellHaswellEP(
                Parameters params) :
                cpu(params), cuda(params)
            {
                #if defined(DEBUG)
                cout << "Maxwell-HaswellEP::" << __func__ << endl;
                cout << params;
                #endif

            }

            /*
                High level routines
                These routines operate on grids
            */
            void MaxwellHaswellEP::grid_visibilities(
                const std::complex<float> *visibilities,
                const float *uvw,
                const float *wavenumbers,
                const int *metadata,
                std::complex<float> *grid,
                const float w_offset,
                const std::complex<float> *aterm,
                const float *spheroidal) {
                #if defined(DEBUG)
                cout << "MaxwellHaswellEP::" << __func__ << endl;
                #endif

                // Load kernels
                kernel::cuda::Gridder kernel_gridder = cuda.get_kernel_gridder();
                kernel::cuda::GridFFT kernel_fft_small = cuda.get_kernel_fft();
                kernel::cpu::GridFFT kernel_fft_big = cpu.get_kernel_fft();
                kernel::cpu::Adder kernel_adder = cpu.get_kernel_adder();
            }

            void MaxwellHaswellEP::degrid_visibilities(
                std::complex<float> *visibilities,
                const float *uvw,
                const float *wavenumbers,
                const int *metadata,
                const std::complex<float> *grid,
                const float w_offset,
                const std::complex<float> *aterm,
                const float *spheroidal) {
                #if defined(DEBUG)
                cout << "MaxwellHaswellEP::" << __func__ << endl;
                #endif
            }

            void MaxwellHaswellEP::transform(DomainAtoDomainB direction,
                std::complex<float>* grid) {
                #if defined(DEBUG)
                cout << "MaxwellHaswellEP::" << __func__ << endl;
                #endif
            }

        } // namespace hybrid
    } // namespace proxy
} // namespace idg
