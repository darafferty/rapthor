#include "Reference.h"

using namespace std;

namespace idg {
    namespace proxy {
        namespace cpu {

            // Constructor
            Reference::Reference(
                std::vector<std::string> libraries)
                : CPU(libraries)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
            }

            std::vector<std::string> Reference::default_libraries() {
                std::string prefix = "Reference/libcpu-reference-kernel-";
                std::vector<std::string> libraries;
                libraries.push_back(prefix + "gridder.so");
                libraries.push_back(prefix + "degridder.so");
                libraries.push_back(prefix + "adder.so");
                libraries.push_back(prefix + "splitter.so");
                libraries.push_back(prefix + "fft.so");
                return libraries;
            }

        } // namespace cpu
    } // namespace proxy
} // namespace idg

#include "ReferenceC.h"
