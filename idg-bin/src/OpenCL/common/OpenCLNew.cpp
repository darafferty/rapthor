#include "OpenCLNew.h"

namespace idg {
    namespace proxy {
        namespace opencl {
            OpenCLNew::OpenCLNew(
                Parameters params) {

                #if defined(DEBUG)
                std::cout << "OPENCL::" << __func__ << std::endl;
                std::cout << params;
                #endif

                mParams = params;
                init_devices();
                print_devices();
                print_compiler_flags();
            }

            void OpenCLNew::init_devices() {

            }

            void OpenCLNew::print_devices() {

            }

            void OpenCLNew::print_compiler_flags() {

            }
        } // namespace opencl
    } // namespace proxy
} // namespace idg
