#ifndef IDG_OPENCL_INSTANCE_H_
#define IDG_OPENCL_INSTANCE_H_

#include <cstring>
#include <sstream>
#include <memory>
#include <fstream>

#include "idg-common.h"

#include "Util.h"


namespace idg {
    namespace kernel {
        namespace opencl {
            typedef size_t clfftPlanHandle;
            class PowerRecord;

            class InstanceOpenCL : public KernelsInstance {
                public:
                    // Constructor
                    InstanceOpenCL(
                        cl::Context& context,
                        int device_nr,
                        int device_id);

                    // Destructor
                    ~InstanceOpenCL();

                    cl::Device&       get_device()  const { return *device; }
                    cl::CommandQueue& get_execute_queue() const { return *executequeue; };
                    cl::CommandQueue& get_htod_queue() const { return *htodqueue; };
                    cl::CommandQueue& get_dtoh_queue() const { return *dtohqueue; };

                    std::string get_compiler_flags();

                void launch_gridder(
                    int nr_timesteps,
                    int nr_subgrids,
                    int grid_size,
                    int subgrid_size,
                    float image_size,
                    float w_step,
                    int nr_channels,
                    int nr_stations,
                    cl::Buffer& d_uvw,
                    cl::Buffer& d_wavenumbers,
                    cl::Buffer& d_visibilities,
                    cl::Buffer& d_spheroidal,
                    cl::Buffer& d_aterm,
                    cl::Buffer& d_metadata,
                    cl::Buffer& d_subgrid);

                void launch_degridder(
                    int nr_timesteps,
                    int nr_subgrids,
                    int grid_size,
                    int subgrid_size,
                    float image_size,
                    float w_step,
                    int nr_channels,
                    int nr_stations,
                    cl::Buffer& d_uvw,
                    cl::Buffer& d_wavenumbers,
                    cl::Buffer& d_visibilities,
                    cl::Buffer& d_spheroidal,
                    cl::Buffer& d_aterm,
                    cl::Buffer& d_metadata,
                    cl::Buffer& d_subgrid);

                void plan_fft(
                    unsigned size, unsigned batch);

                void launch_fft(
                    cl::Buffer &d_data,
                    DomainAtoDomainB direction);

                void launch_adder(
                    int nr_subgrids,
                    int grid_size,
                    int subgrid_size,
                    cl::Buffer& d_metadata,
                    cl::Buffer& d_subgrid,
                    cl::Buffer& d_grid);

                void launch_splitter(
                    int nr_subgrids,
                    int grid_size,
                    int subgrid_size,
                    cl::Buffer& d_metadata,
                    cl::Buffer& d_subgrid,
                    cl::Buffer& d_grid);

                void launch_scaler(
                    int nr_subgrids,
                    int subgrid_size,
                    cl::Buffer& d_subgrid);

                protected:
                    void compile_kernel_gridder(unsigned nr_channels);
                    void compile_kernel_degridder(unsigned nr_channels);
                    cl::Kernel* compile_kernel(
                        int kernel_id,
                        std::string file_name,
                        std::string kernel_name,
                        std::string flags_misc = "");
                    void set_parameters();
                    void set_parameters_default();
                    void set_parameters_vega();
                    void set_parameters_fiji();
                    void set_parameters_hawaii();
                    void set_parameters_tahiti();

                private:
                    cl::Context& mContext;
                    cl::Device *device;
                    cl::CommandQueue  *executequeue;
                    cl::CommandQueue  *htodqueue;
                    cl::CommandQueue  *dtohqueue;
                    cl::Kernel *kernel_gridder;
                    cl::Kernel *kernel_degridder;
                    cl::Kernel *kernel_adder;
                    cl::Kernel *kernel_splitter;
                    cl::Kernel *kernel_scaler;
                    std::vector<cl::Program*> mPrograms;

                protected:
                    cl::NDRange block_gridder;
                    cl::NDRange block_degridder;
                    cl::NDRange block_adder;
                    cl::NDRange block_splitter;
                    cl::NDRange block_scaler;

                    // (De)gridder kernel
                    int nr_channels_gridder;
                    int nr_channels_degridder;
                    int batch_gridder;
                    int batch_degridder;

                    // FFT kernel
                    bool fft_planned;
                    unsigned int fft_planned_size;
                    unsigned int fft_planned_batch;
                    clfftPlanHandle fft_plan;

                public:
                    // Memory management
                    cl::Buffer& get_device_grid(
                        unsigned int grid_size = 0);

                    cl::Buffer& get_device_wavenumbers(
                        unsigned int nr_channels = 0);

                    cl::Buffer& get_device_aterms(
                        unsigned int nr_stations  = 0,
                        unsigned int nr_timeslots = 0,
                        unsigned int subgrid_size = 0);

                    cl::Buffer& get_device_spheroidal(
                        unsigned int subgrid_size = 0);

                    cl::Buffer& get_host_grid(
                        unsigned int grid_size = 0);

                    cl::Buffer& get_host_visibilities(
                        unsigned int nr_baselines = 0,
                        unsigned int nr_timesteps = 0,
                        unsigned int nr_channels = 0,
                        void *ptr = NULL);

                    cl::Buffer& get_host_uvw(
                        unsigned int nr_baselines = 0,
                        unsigned int nr_timesteps = 0,
                        void *ptr = NULL);

                public:
                    powersensor::State measure();
                    void measure(PowerRecord &record, cl::CommandQueue &queue);

                    void enqueue_report(
                        cl::CommandQueue &queue,
                        int nr_timesteps,
                        int nr_subgrids);

                private:
                    void start_measurement(void *data);
                    void end_measurement(void *data);


                private:
                    cl::Buffer* reuse_memory(
                        uint64_t size,
                        cl::Buffer *buffer,
                        cl_mem_flags flags,
                        void *ptr = NULL);

                    cl::Buffer *d_grid;
                    cl::Buffer *d_wavenumbers;
                    cl::Buffer *d_aterms;
                    cl::Buffer *d_spheroidal;
                    cl::Buffer *h_grid;
                    cl::Buffer *h_visibilities;
                    cl::Buffer *h_uvw;

            };

            std::ostream& operator<<(std::ostream& os, InstanceOpenCL &d);

			// Kernel names
			static const std::string file_gridder   = "KernelGridder.cl";
			static const std::string file_degridder = "KernelDegridder.cl";
			static const std::string file_adder     = "KernelAdder.cl";
			static const std::string file_splitter  = "KernelSplitter.cl";
			static const std::string file_scaler    = "KernelScaler.cl";
			static const std::string name_gridder   = "kernel_gridder";
			static const std::string name_degridder = "kernel_degridder";
			static const std::string name_adder     = "kernel_adder";
			static const std::string name_splitter  = "kernel_splitter";
			static const std::string name_scaler    = "kernel_scaler";

        } // end namespace opencl
    } // end namespace kernel
} // end namespace idg

#endif
