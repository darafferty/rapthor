#ifndef IDG_KERNELS_H_
#define IDG_KERNELS_H_

#include <cstdint>
#include <ccomplex>
#include <cmath>
#include <string>

#include "../../common/RuntimeWrapper.h"
#include "../../common/Parameters.h"

namespace idg {
    namespace kernel {
        namespace cpu {

            // define the kernel function names
            static const std::string name_gridder = "kernel_gridder";
            static const std::string name_degridder = "kernel_degridder";
            static const std::string name_fft = "kernel_fft";
            static const std::string name_adder = "kernel_adder";
            static const std::string name_splitter = "kernel_splitter";

            // Function signatures
            #define sig_gridder   (void (*)(int,float,void*,void*,void*,void*,void*,void*,void*))
            #define sig_degridder (void (*)(int,float,void*,void*,void*,void*,void*,void*,void*))
            #define sig_fft		  (void (*)(int,int,void*,int))
            #define sig_adder	  (void (*)(int,void*,void*,void*))
            #define sig_splitter  (void (*)(int,void*,void*,void*))

            // define auxiliary function names
            static const std::string name_gridder_flops = "kernel_gridder_flops";
            static const std::string name_degridder_flops = "kernel_degridder_flops";
            static const std::string name_fft_flops = "kernel_fft_flops";
            static const std::string name_adder_flops = "kernel_adder_flops";
            static const std::string name_splitter_flops = "kernel_splitter_flops";

            static const std::string name_gridder_bytes = "kernel_gridder_bytes";
            static const std::string name_degridder_bytes = "kernel_degridder_bytes";
            static const std::string name_fft_bytes = "kernel_fft_bytes";
            static const std::string name_adder_bytes = "kernel_adder_bytes";
            static const std::string name_splitter_bytes = "kernel_splitter_bytes";

            class Gridder {
                public:
                    Gridder(runtime::Module &module, const Parameters &parameters);
                    void run(int jobsize, float w_offset, void *uvw, void *wavenumbers,
                             void *visibilities, void *spheroidal, void *aterm,
                             void *metadata, void *subgrid);
                    uint64_t flops(int jobsize, int nr_subgrids);
                    uint64_t bytes(int jobsize, int nr_subgrids);

                private:
                    runtime::Function _run;
                    Parameters parameters;
            };


            class Degridder {
                public:
                    Degridder(runtime::Module &module, const Parameters &parameters);
                    void run(int jobsize, float w_offset, void *uvw, void *wavenumbers,
                             void *visibilities, void *spheroidal, void *aterm,
                             void *metadata, void *subgrid);
                    uint64_t flops(int jobsize, int nr_subgrids);
                    uint64_t bytes(int jobsize, int nr_subgrids);

                private:
                    runtime::Function _run;
                    Parameters parameters;
            };


            class GridFFT {
                public:
                    GridFFT(runtime::Module &module, const Parameters &parameters);
                    void run(int size, int batch, void *data, int direction);
                    uint64_t flops(int size, int batch);
                    uint64_t bytes(int size, int batch);

                private:
                    runtime::Function _run;
                    Parameters parameters;
            };


            class Adder {
                public:
                    Adder(runtime::Module &module, const Parameters &parameters);
                    void run(int jobsize, void *metadata, void *subgrid, void *grid);
                    uint64_t flops(int nr_subgrids);
                    uint64_t bytes(int nr_subgrids);

                private:
                    runtime::Function _run;
                    Parameters parameters;
            };


            class Splitter {
                public:
                    Splitter(runtime::Module &module, const Parameters &parameters);
                    void run(int jobsize, void *metadata, void *subgrid, void *grid);
                    uint64_t flops(int nr_subgrids);
                    uint64_t bytes(int nr_subgrids);

                private:
                    runtime::Function _run;
                    Parameters parameters;
            };

        } // namespace cpu
    } // namespace kernel
} // namespace idg

#endif
