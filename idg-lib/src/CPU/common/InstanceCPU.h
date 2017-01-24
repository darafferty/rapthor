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

            class Gridder {
                public:
                    Gridder(runtime::Module &module);
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
            };


            class Degridder {
                public:
                    Degridder(runtime::Module &module);
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
            };


            class GridFFT {
                public:
                    GridFFT(runtime::Module &module);
                    void run(
                        int gridsize,
                        int size,
                        int batch,
                        void *data,
                        int direction);

                private:
                    runtime::Function _run;
            };


            class Adder {
                public:
                    Adder(runtime::Module &module);
                    void run(
                        int nr_subgrids,
                        int gridsize,
                        void *metadata,
                        void *subgrid,
                        void *grid);

                private:
                    runtime::Function _run;
            };


            class Splitter {
                public:
                    Splitter(runtime::Module &module);
                    void run(
                        int nr_subgrids,
                        int gridsize,
                        void *metadata,
                        void *subgrid,
                        void *grid);

                private:
                    runtime::Function _run;
            };


            class InstanceCPU : public Kernels
            {
                public:
                    InstanceCPU(
                        CompileConstants constants,
                        Compiler compiler,
                        Compilerflags flags,
                        ProxyInfo info);

                    virtual ~InstanceCPU();

                    virtual std::unique_ptr<idg::kernel::cpu::Gridder> get_kernel_gridder() const;
                    virtual std::unique_ptr<idg::kernel::cpu::Degridder> get_kernel_degridder() const;
                    virtual std::unique_ptr<idg::kernel::cpu::Adder> get_kernel_adder() const;
                    virtual std::unique_ptr<idg::kernel::cpu::Splitter> get_kernel_splitter() const;
                    virtual std::unique_ptr<idg::kernel::cpu::GridFFT> get_kernel_fft() const;


                    static std::string make_tempdir();
                    static ProxyInfo default_proxyinfo(
                        std::string srcdir,
                        std::string tmpdir);

                protected:
                    void compile(
                        Compiler compiler,
                        Compilerflags flags);
                    void load_shared_objects();
                    void find_kernel_functions();

                    ProxyInfo mInfo;

                    // Data structures to find out in which .so-file a kernel is defined
                    std::vector<runtime::Module*> modules;
                    std::map<std::string,int> which_module;
            };

        } // namespace cpu
    } // namespace kernel
} // namespace idg

#endif
