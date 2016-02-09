#include <complex>
#include <sstream>
#include <memory>
#include <omp.h> // omp_get_wtime

#include "idg-config.h"
#include "Reference.h"

using namespace std;
using namespace idg::kernel::opencl;

namespace idg {
    namespace proxy {
        namespace opencl {

            /// Constructors
            Reference::Reference(
                Parameters params,
                unsigned deviceNumber,
                Compilerflags flags)
                : OpenCL(params, deviceNumber, flags)
            {
                #if defined(DEBUG)
                cout << "Reference::" << __func__ << endl;
                cout << "Compiler flags: " << flags << endl;
                cout << params;
                #endif
            }

            string Reference::default_compiler_flags() {
                return OpenCL::default_compiler_flags();
            }

            unique_ptr<Gridder> Reference::get_kernel_gridder() const {
                return unique_ptr<Gridder>(new Gridder(*(programs[which_program.at(name_gridder)]), mParams));
            }

            unique_ptr<Degridder> Reference::get_kernel_degridder() const {
                return unique_ptr<Degridder>(new Degridder(*(programs[which_program.at(name_degridder)]), mParams));
            }

            unique_ptr<GridFFT> Reference::get_kernel_fft() const {
                return unique_ptr<GridFFT>(new GridFFT(mParams));
            }

            unique_ptr<Scaler> Reference::get_kernel_scaler() const {
                return unique_ptr<Scaler>(new Scaler(*(programs[which_program.at(name_scaler)]), mParams));
            }

            unique_ptr<Adder> Reference::get_kernel_adder() const {
                return unique_ptr<Adder>(new Adder(*(programs[which_program.at(name_adder)]), mParams));
            }

            unique_ptr<Splitter> Reference::get_kernel_splitter() const {
                return unique_ptr<Splitter>(new Splitter(*(programs[which_program.at(name_splitter)]), mParams));
            }

        } // namespace opencl
    } // namespace proxy
} // namespace idg

// C interface:
// Rationale: calling the code from C code and Fortran easier,
// and bases to create interface to scripting languages such as
// Python, Julia, Matlab, ...
extern "C" {
    typedef idg::proxy::opencl::Reference OpenCL_Reference;

    OpenCL_Reference* OpenCL_Reference_init(
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

        return new OpenCL_Reference(P);
    }

    void OpenCL_Reference_grid(OpenCL_Reference* p,
                            void *visibilities,
                            void *uvw,
                            void *wavenumbers,
                            void *metadata,
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
                (const int*) metadata,
                (std::complex<float>*) grid,
                w_offset,
                kernel_size,
                (const std::complex<float>*) aterm,
                (const int*) aterm_offsets,
                (const float*) spheroidal);
    }

    void OpenCL_Reference_degrid(OpenCL_Reference* p,
                            void *visibilities,
                            void *uvw,
                            void *wavenumbers,
                            void *metadata,
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
                    (const int*) metadata,
                    (const std::complex<float>*) grid,
                    w_offset,
                    kernel_size,
                    (const std::complex<float>*) aterm,
                    (const int*) aterm_offsets,
                    (const float*) spheroidal);
     }

    void OpenCL_Reference_transform(OpenCL_Reference* p,
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

    void OpenCL_Reference_destroy(OpenCL_Reference* p) {
       delete p;
    }

}  // end extern "C"
