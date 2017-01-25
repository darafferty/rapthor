#ifndef IDG_KERNELS_CPU_H_
#define IDG_KERNELS_CPU_H_

#include <cstdint>
#include <ccomplex>
#include <cmath>
#include <string>
#include <memory> // unique_ptr

#include "idg-common.h"

namespace idg {
    namespace kernel {
        namespace cpu {

            class InstanceCPU : public KernelsInstance
            {
                public:
                    // Constructor
                    InstanceCPU(
                        CompileConstants constants,
                        Compiler compiler,
                        Compilerflags flags,
                        ProxyInfo info);

                    // Destructor
                    virtual ~InstanceCPU();

                    void run_gridder(
                        int nr_subgrids,
                        int gridsize,
                        float image_size,
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

                    void run_degridder(
                        int nr_subgrids,
                        int gridsize,
                        float image_size,
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

                    void run_fft(
                        int gridsize,
                        int size,
                        int batch,
                        void *data,
                        int direction);

                    void run_adder(
                        int nr_subgrids,
                        int gridsize,
                        void *metadata,
                        void *subgrid,
                        void *grid);

                    void run_splitter(
                        int nr_subgrids,
                        int gridsize,
                        void *metadata,
                        void *subgrid,
                        void *grid);

                    static std::string make_tempdir();
                    static ProxyInfo default_proxyinfo(
                        std::string srcdir,
                        std::string tmpdir);

                protected:
                    void compile(
                        Compiler compiler,
                        Compilerflags flags);
                    void load_shared_objects();
                    void load_kernel_funcions();

                    ProxyInfo mInfo;
                    std::vector<runtime::Module*> modules;

                    runtime::Function *function_gridder;
                    runtime::Function *function_degridder;
                    runtime::Function *function_fft;
                    runtime::Function *function_adder;
                    runtime::Function *function_splitter;

            };

            // Kernel names
            static const std::string name_gridder   = "kernel_gridder";
            static const std::string name_degridder = "kernel_degridder";
            static const std::string name_adder     = "kernel_adder";
            static const std::string name_splitter  = "kernel_splitter";
            static const std::string name_fft       = "kernel_fft";
            static const std::string name_scaler    = "kernel_scaler";

        } // end namespace cpu
    } // end namespace kernel
} // end namespace idg

#endif
