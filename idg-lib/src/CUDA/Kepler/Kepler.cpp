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
                ProxyInfo info) :
                CUDA(params, deviceNumber, info)
            {
                #if defined(DEBUG)
                cout << "Kepler::" << __func__ << endl;
                cout << "Compiler: " << compiler << endl;
                cout << "Compiler flags: " << flags << endl;
                cout << params;
                #endif

                init_cuda(deviceNumber);
                compile_kernels(compiler, append(flags));
                init_powersensor();
            }

            dim3 Kepler::get_block_gridder() const {
                return dim3(16, 16);
            }

            dim3 Kepler::get_block_degridder() const {
                return dim3(128);
            }

            dim3 Kepler::get_block_adder() const {
                return dim3(128);
            }

            dim3 Kepler::get_block_splitter() const {
                return dim3(128);
            }

            dim3 Kepler::get_block_scaler() const {
                return dim3(128);
            }

            int Kepler::get_gridder_batch_size() const {
                return 32;
            }

            int Kepler::get_degridder_batch_size() const {
                dim3 block_degridder = get_block_degridder();
                return block_degridder.x * block_degridder.y * block_degridder.z;
            }

            Compilerflags Kepler::append(Compilerflags flags) const {
                stringstream new_flags;
                new_flags << flags;
                new_flags << " -DGRIDDER_BATCH_SIZE=" << get_gridder_batch_size();
                new_flags << " -DDEGRIDDER_BATCH_SIZE=" << get_degridder_batch_size();
                return new_flags.str();
            }
        } // namespace cuda
    } // namespace proxy
} // namespace idg
