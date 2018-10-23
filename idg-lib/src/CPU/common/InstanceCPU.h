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
                        std::vector<std::string> libraries);

                    // Destructor
                    virtual ~InstanceCPU();

                    void run_gridder(
                        int nr_subgrids,
                        int grid_size,
                        int subgrid_size,
                        float image_size,
                        float w_step,
                        const float* shift,
                        int nr_channels,
                        int nr_stations,
                        void *uvw,
                        void *wavenumbers,
                        void *visibilities,
                        void *spheroidal,
                        void *aterm,
                        void *avg_aterm,
                        void *metadata,
                        void *subgrid);

                    void run_degridder(
                        int nr_subgrids,
                        int grid_size,
                        int subgrid_size,
                        float image_size,
                        float w_step,
                        const float* shift,
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
                        int grid_size,
                        int size,
                        int batch,
                        void *data,
                        int direction);

                     void run_subgrid_fft(
                        int grid_size,
                        int size,
                        int batch,
                        void *data,
                        int direction);

                    void run_adder(
                        int nr_subgrids,
                        int grid_size,
                        int subgrid_size,
                        void *metadata,
                        void *subgrid,
                        void *grid);

                    void run_splitter(
                        int nr_subgrids,
                        int grid_size,
                        int subgrid_size,
                        void *metadata,
                        void *subgrid,
                        void *grid);
                    
                    void run_adder_wstack(
                        int nr_subgrids,
                        int grid_size,
                        int subgrid_size,
                        void *metadata,
                        void *subgrid,
                        void *grid);

                    void run_splitter_wstack(
                        int nr_subgrids,
                        int grid_size,
                        int subgrid_size,
                        void *metadata,
                        void *subgrid,
                        void *grid);
                    
                    bool has_adder_wstack() {return (function_adder_wstack != nullptr);}

                    bool has_splitter_wstack() {return (function_splitter_wstack != nullptr);}

                protected:
                    void compile(
                        Compiler compiler,
                        Compilerflags flags);
                    void load_shared_objects(
                        std::vector<std::string> libraries);
                    void load_kernel_funcions();

                    std::vector<runtime::Module*> modules;

                    runtime::Function *function_gridder;
                    runtime::Function *function_degridder;
                    runtime::Function *function_fft;
                    runtime::Function *function_adder;
                    runtime::Function *function_splitter;
                    runtime::Function *function_adder_wstack;
                    runtime::Function *function_splitter_wstack;
            };

            // Jobsize
            static const int jobsize_gridding   = 1024;
            static const int jobsize_degridding = 1024;

            // Kernel names
            static const std::string name_gridder         = "kernel_gridder";
            static const std::string name_degridder       = "kernel_degridder";
            static const std::string name_adder           = "kernel_adder";
            static const std::string name_splitter        = "kernel_splitter";
            static const std::string name_adder_wstack    = "kernel_adder_wstack";
            static const std::string name_splitter_wstack = "kernel_splitter_wstack";
            static const std::string name_fft             = "kernel_fft";
            static const std::string name_scaler          = "kernel_scaler";

        } // end namespace cpu
    } // end namespace kernel
} // end namespace idg

#endif
