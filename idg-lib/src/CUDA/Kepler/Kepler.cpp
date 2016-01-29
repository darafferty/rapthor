#include "Kepler.h"

using namespace std;
using namespace idg::kernel::cuda;

namespace idg {
    namespace proxy {
        namespace cuda {
            /// Constructors
            Kepler::Kepler(
                Parameters params,
                unsigned deviceNumber,
                Compiler compiler,
                Compilerflags flags,
                ProxyInfo info)
                : CUDA(params, deviceNumber, compiler, append(flags), info)
            {
                #if defined(DEBUG)
                cout << "Kepler::" << __func__ << endl;
                cout << "Compiler: " << compiler << endl;
                cout << "Compiler flags: " << flags << endl;
                cout << params;
                #endif
            }

            Compilerflags Kepler::append(Compilerflags flags) {
                stringstream new_flags;
                new_flags << flags;
                new_flags << " -DMAX_NR_TIMESTEPS_GRIDDER=" << 32;
                new_flags << " -DMAX_NR_TIMESTEPS_DEGRIDDER=" << 64;
                return new_flags.str();
            }

            unique_ptr<Gridder> Kepler::get_kernel_gridder() const {
                return unique_ptr<Gridder>(new GridderKepler(*(modules[which_module.at(name_gridder)]), mParams));
            }

            unique_ptr<Degridder> Kepler::get_kernel_degridder() const {
                return unique_ptr<Degridder>(new DegridderKepler(*(modules[which_module.at(name_degridder)]), mParams));
            }

            unique_ptr<GridFFT> Kepler::get_kernel_fft() const {
                return unique_ptr<GridFFT>(new GridFFTKepler(*(modules[which_module.at(name_fft)]), mParams));
            }

            unique_ptr<Scaler> Kepler::get_kernel_scaler() const {
                return unique_ptr<Scaler>(new ScalerKepler(*(modules[which_module.at(name_scaler)]), mParams));
            }

            unique_ptr<Adder> Kepler::get_kernel_adder() const {
                return unique_ptr<Adder>(new AdderKepler(*(modules[which_module.at(name_adder)]), mParams));
            }

            unique_ptr<Splitter> Kepler::get_kernel_splitter() const {
                return unique_ptr<Splitter>(new SplitterKepler(*(modules[which_module.at(name_splitter)]), mParams));
            }
        } // namespace cuda
    } // namespace proxy
} // namespace idg
