#include "Maxwell.h"

using namespace std;
using namespace idg::kernel::cuda;

namespace idg {
    namespace proxy {
        namespace cuda {
            /// Constructors
            Maxwell::Maxwell(
                Parameters params,
                unsigned deviceNumber,
                Compiler compiler,
                Compilerflags flags,
                ProxyInfo info)
                : CUDA( params, deviceNumber, compiler, append(flags), info)
            {
                #if defined(DEBUG)
                cout << "Maxwell::" << __func__ << endl;
                cout << "Compiler: " << compiler << endl;
                cout << "Compiler flags: " << flags << endl;
                cout << params;
                #endif
            }

            Compilerflags Maxwell::append(Compilerflags flags) {
                stringstream new_flags;
                new_flags << flags;
                new_flags << " -DMAX_NR_TIMESTEPS_GRIDDER=" << GridderMaxwell::max_nr_timesteps;
                new_flags << " -DMAX_NR_TIMESTEPS_DEGRIDDER=" << DegridderMaxwell::max_nr_timesteps;
                new_flags << " -DNR_THREADS_DEGRIDDER=" << DegridderMaxwell::nr_threads;
                return new_flags.str();
            }

            unique_ptr<Gridder> Maxwell::get_kernel_gridder() const {
                return unique_ptr<Gridder>(new GridderMaxwell(*(modules[which_module.at(name_gridder)]), mParams));
            }

            unique_ptr<Degridder> Maxwell::get_kernel_degridder() const {
                return unique_ptr<Degridder>(new DegridderMaxwell(*(modules[which_module.at(name_degridder)]), mParams));
            }

            unique_ptr<GridFFT> Maxwell::get_kernel_fft() const {
                return unique_ptr<GridFFT>(new GridFFTMaxwell(*(modules[which_module.at(name_fft)]), mParams));
            }

            unique_ptr<Scaler> Maxwell::get_kernel_scaler() const {
                return unique_ptr<Scaler>(new ScalerMaxwell(*(modules[which_module.at(name_scaler)]), mParams));
            }

            unique_ptr<Adder> Maxwell::get_kernel_adder() const {
                return unique_ptr<Adder>(new AdderMaxwell(*(modules[which_module.at(name_adder)]), mParams));
            }

            unique_ptr<Splitter> Maxwell::get_kernel_splitter() const {
                return unique_ptr<Splitter>(new SplitterMaxwell(*(modules[which_module.at(name_splitter)]), mParams));
            }
        } // namespace cuda
    } // namespace proxy
} // namespace idg



// C interface:
// Rationale: calling the code from C code and Fortran easier,
// and bases to create interface to scripting languages such as
// Python, Julia, Matlab, ...
extern "C" {
    typedef idg::proxy::cuda::Maxwell CUDA_Maxwell;

    CUDA_Maxwell* CUDA_Maxwell_init(
                unsigned int nr_stations,
                unsigned int nr_channels,
                unsigned int nr_time,
                unsigned int nr_timeslots,
                float        imagesize,
                unsigned int grid_size,
                unsigned int subgrid_size)
    {
        idg::Parameters P;
        P.set_nr_stations(nr_stations);
        P.set_nr_channels(nr_channels);
        P.set_nr_time(nr_time);
        P.set_nr_timeslots(nr_timeslots);
        P.set_imagesize(imagesize);
        P.set_subgrid_size(subgrid_size);
        P.set_grid_size(grid_size);

        return new CUDA_Maxwell(P);
    }

    void CUDA_Maxwell_grid(CUDA_Maxwell* p,
                            void *visibilities,
                            void *uvw,
                            void *wavenumbers,
                            void *baselines,
                            void *grid,
                            float w_offset,
                            int   kernel_size,
                            void *aterm,
                            void *aterm_offsets,
                            void *spheroidal)
    {
         p->grid_visibilities(
                (const std::complex<float>*) visibilities,
                (const float*) uvw,
                (const float*) wavenumbers,
                (const int*) baselines,
                (std::complex<float>*) grid,
                w_offset,
                kernel_size,
                (const std::complex<float>*) aterm,
                (const int*) aterm_offsets,
                (const float*) spheroidal);
    }

    void CUDA_Maxwell_degrid(CUDA_Maxwell* p,
                            void *visibilities,
                            void *uvw,
                            void *wavenumbers,
                            void *baselines,
                            void *grid,
                            float w_offset,
                            int   kernel_size,
                            void *aterm,
                            void *aterm_offsets,
                            void *spheroidal)
    {
         p->degrid_visibilities(
                (std::complex<float>*) visibilities,
                (const float*) uvw,
                (const float*) wavenumbers,
                (const int*) baselines,
                (const std::complex<float>*) grid,
                w_offset,
                kernel_size,
                (const std::complex<float>*) aterm,
                (const int*) aterm_offsets,
                (const float*) spheroidal);
    }

    void CUDA_Maxwell_transform(CUDA_Maxwell* p,
                    int direction,
                    void *grid)
    {
       if (direction!=0)
           p->transform(idg::ImageDomainToFourierDomain,
                    (std::complex<float>*) grid);
       else
           p->transform(idg::FourierDomainToImageDomain,
                    (std::complex<float>*) grid);
    }

    void CUDA_Maxwell_destroy(CUDA_Maxwell* p) {
       delete p;
    }
}  // end extern "C"
