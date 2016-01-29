#include "idg-config.h"
#include "Jetson.h"

using namespace std;
using namespace idg::kernel::cuda;

namespace idg {
    namespace proxy {
        namespace cuda {
            /// Constructors
            Jetson::Jetson(
                Parameters params,
                unsigned deviceNumber,
                Compiler compiler,
                Compilerflags flags,
                ProxyInfo info)
                : CUDA(params, deviceNumber, compiler, append(flags), info)
            {
                #if defined(DEBUG)
                cout << "Jetson::" << __func__ << endl;
                cout << "Compiler: " << compiler << endl;
                cout << "Compiler flags: " << flags << endl;
                cout << params;
                #endif
            }

            Compilerflags Jetson::append(Compilerflags flags) {
                stringstream new_flags;
                new_flags << flags;
                new_flags << " -DMAX_NR_TIMESTEPS_GRIDDER=" << GridderMaxwell::max_nr_timesteps;
                new_flags << " -DMAX_NR_TIMESTEPS_DEGRIDDER=" << DegridderMaxwell::max_nr_timesteps;
                new_flags << " -DNR_THREADS_DEGRIDDER=" << DegridderMaxwell::nr_threads;
                return new_flags.str();
            }

            void Jetson::transform(DomainAtoDomainB direction,
                                complex<float>* grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                cout << "Transform direction: " << direction << endl;
                #endif
            }

            void Jetson::grid_visibilities(
                const std::complex<float> *visibilities,
                const float *uvw,
                const float *wavenumbers,
                const int *baselines,
                std::complex<float> *grid,
                const float w_offset,
                const int kernel_size,
                const std::complex<float> *aterm,
                const int *aterm_offsets,
                const float *spheroidal)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
                cout << "Not implemented" << endl;
            }

            void Jetson::degrid_visibilities(
                std::complex<float> *visibilities,
                const float *uvw,
                const float *wavenumbers,
                const int *baselines,
                const std::complex<float> *grid,
                const float w_offset,
                const int kernel_size,
                const std::complex<float> *aterm,
                const int *aterm_offsets,
                const float *spheroidal)
             {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
                cout << "Not implemented" << endl;
            }
 
            unique_ptr<Gridder> Jetson::get_kernel_gridder() const {
                return unique_ptr<Gridder>(new GridderJetson(*(modules[which_module.at(name_gridder)]), mParams));
            }

            unique_ptr<Degridder> Jetson::get_kernel_degridder() const {
                return unique_ptr<Degridder>(new DegridderJetson(*(modules[which_module.at(name_degridder)]), mParams));
            }

            unique_ptr<GridFFT> Jetson::get_kernel_fft() const {
                return unique_ptr<GridFFT>(new GridFFTJetson(*(modules[which_module.at(name_fft)]), mParams));
            }

            unique_ptr<Scaler> Jetson::get_kernel_scaler() const {
                return unique_ptr<Scaler>(new ScalerJetson(*(modules[which_module.at(name_scaler)]), mParams));
            }

            unique_ptr<Adder> Jetson::get_kernel_adder() const {
                return unique_ptr<Adder>(new AdderJetson(*(modules[which_module.at(name_adder)]), mParams));
            }

            unique_ptr<Splitter> Jetson::get_kernel_splitter() const {
                return unique_ptr<Splitter>(new SplitterJetson(*(modules[which_module.at(name_splitter)]), mParams));
            }
        } // namespace cuda
    } // namespace proxy
} // namespace idg
