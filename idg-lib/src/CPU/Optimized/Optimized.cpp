#include "Optimized.h"

using namespace std;

namespace idg {
    namespace proxy {
        namespace cpu {

            // Constructor
            Optimized::Optimized(
                std::vector<std::string> libraries)
                : CPU(libraries)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
            }

            std::vector<std::string> Optimized::default_libraries() {
                std::string prefix = "Optimized/libcpu-optimized-kernel-";
                std::vector<std::string> libraries;
                libraries.push_back(prefix + "gridder.so");
                libraries.push_back(prefix + "degridder.so");
                libraries.push_back(prefix + "calibrate.so");
                libraries.push_back(prefix + "adder.so");
                libraries.push_back(prefix + "splitter.so");
                libraries.push_back(prefix + "fft.so");
                libraries.push_back(prefix + "adder-wstack.so");
                libraries.push_back(prefix + "splitter-wstack.so");
                libraries.push_back(prefix + "adder-wtiles.so");
                libraries.push_back(prefix + "splitter-wtiles.so");
                return libraries;
            }

        } // namespace cpu
    } // namespace proxy
} // namespace idg


#include "OptimizedC.h"
