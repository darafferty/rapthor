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
