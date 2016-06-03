#include "Maxwell.h"

using namespace std;
using namespace idg::kernel::cuda;

namespace idg {
    namespace proxy {
        namespace cuda {
            Maxwell::Maxwell(
                Parameters params,
                unsigned deviceNumber,
                Compiler compiler,
                Compilerflags flags,
                ProxyInfo info) :
                CUDA(params, deviceNumber, info)
            {
                #if defined(DEBUG)
                cout << "Maxwell::" << __func__ << endl;
                cout << "Compiler: " << compiler << endl;
                cout << "Compiler flags: " << flags << endl;
                cout << params;
                #endif

                init_cuda(deviceNumber);
                compile_kernels(compiler, append(flags));
                init_powersensor();
            }

            dim3 Maxwell::get_block_gridder() const {
                return dim3(32, 4);
            }

            dim3 Maxwell::get_block_degridder() const {
                return dim3(128);
            }

            dim3 Maxwell::get_block_adder() const {
                return dim3(128);
            }

            dim3 Maxwell::get_block_splitter() const {
                return dim3(128);
            }

            dim3 Maxwell::get_block_scaler() const {
                return dim3(128);
            }

            int Maxwell::get_gridder_batch_size() const {
                return 64;
            }

            int Maxwell::get_degridder_batch_size() const {
                dim3 block_degridder = get_block_degridder();
                return block_degridder.x * block_degridder.y * block_degridder.z;
            }

            Compilerflags Maxwell::append(Compilerflags flags) const {
                stringstream new_flags;
                new_flags << flags;
                new_flags << " -DGRIDDER_BATCH_SIZE=" << get_gridder_batch_size();
                new_flags << " -DDEGRIDDER_BATCH_SIZE=" << get_degridder_batch_size();
                return new_flags.str();
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
