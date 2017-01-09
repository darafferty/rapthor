#ifndef IDG_KERNELS_CPU_H_
#define IDG_KERNELS_CPU_H_

#include <cstdint>
#include <ccomplex>
#include <cmath>
#include <string>

#include "../../common/RuntimeWrapper.h"
#include "../../common/Parameters.h"

namespace idg {
    namespace kernel {
        namespace cpu {
            // Kernel names
            static const std::string name_gridder   = "kernel_gridder";
            static const std::string name_degridder = "kernel_degridder";
            static const std::string name_adder     = "kernel_adder";
            static const std::string name_splitter  = "kernel_splitter";
            static const std::string name_fft       = "kernel_fft";
            static const std::string name_scaler    = "kernel_scaler";

            // Function signatures
            #define sig_gridder   (void (*)(int,int,float,float,int,int,void*,void*,void*,void*,void*,void*,void*))
            #define sig_degridder (void (*)(int,int,float,float,int,int,void*,void*,void*,void*,void*,void*,void*))
            #define sig_fft		  (void (*)(int,int,int,void*,int))
            #define sig_adder	  (void (*)(int,int,void*,void*,void*))
            #define sig_splitter  (void (*)(int,int,void*,void*,void*))

            // define auxiliary function names
            //static const std::string name_gridder_flops = "kernel_gridder_flops";
            //static const std::string name_degridder_flops = "kernel_degridder_flops";
            //static const std::string name_fft_flops = "kernel_fft_flops";
            //static const std::string name_adder_flops = "kernel_adder_flops";
            //static const std::string name_splitter_flops = "kernel_splitter_flops";

            //static const std::string name_gridder_bytes = "kernel_gridder_bytes";
            //static const std::string name_degridder_bytes = "kernel_degridder_bytes";
            //static const std::string name_fft_bytes = "kernel_fft_bytes";
            //static const std::string name_adder_bytes = "kernel_adder_bytes";
            //static const std::string name_splitter_bytes = "kernel_splitter_bytes";

            class Gridder {
                public:
                    Gridder(runtime::Module &module, const Parameters &parameters);
                    void run(
                        int nr_subgrids,
                        int gridsize,
                        float imagesize,
                        float w_offset,
                        int nr_channels,
                        int nr_stations,
                        void *uvw,
                        void *wavenumbers,
                        void *visibilities,
                        void *spheroidal,
                        void *aterm,
                        void *metadata,
                        void *subgrid);

                private:
                    runtime::Function _run;
                    Parameters parameters;
            };


            class Degridder {
                public:
                    Degridder(runtime::Module &module, const Parameters &parameters);
                    void run(
                        int nr_subgrids,
                        int gridsize,
                        float imagesize,
                        float w_offset,
                        int nr_channels,
                        int nr_stations,
                        void *uvw,
                        void *wavenumbers,
                        void *visibilities,
                        void *spheroidal,
                        void *aterm,
                        void *metadata,
                        void *subgrid);

                private:
                    runtime::Function _run;
                    Parameters parameters;
            };


            class GridFFT {
                public:
                    GridFFT(runtime::Module &module, const Parameters &parameters);
                    void run(
                        int gridsize,
                        int size,
                        int batch,
                        void *data,
                        int direction);

                private:
                    runtime::Function _run;
                    Parameters parameters;
            };


            class Adder {
                public:
                    Adder(runtime::Module &module, const Parameters &parameters);
                    void run(
                        int nr_subgrids,
                        int gridsize,
                        void *metadata,
                        void *subgrid,
                        void *grid);

                private:
                    runtime::Function _run;
                    Parameters parameters;
            };


            class Splitter {
                public:
                    Splitter(runtime::Module &module, const Parameters &parameters);
                    void run(
                        int nr_subgrids,
                        int gridsize,
                        void *metadata,
                        void *subgrid,
                        void *grid);

                private:
                    runtime::Function _run;
                    Parameters parameters;
            };

        } // namespace cpu
    } // namespace kernel
} // namespace idg

#endif
