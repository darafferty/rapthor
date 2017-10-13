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
                        CompileConstants& constants,
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

                    powersensor::PowerSensor* get_powersensor() { return powerSensor; };
                    powersensor::State measure();
                    void measure(PowerRecord &record, cl::CommandQueue &queue);

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
                    int size, int batch);

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
                    powersensor::PowerSensor *powerSensor;

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
                    cl::Buffer* reuse_memory(
                        uint64_t size,
                        cl::Buffer *buffer,
                        cl_mem_flags flags)
                    {
                        if (buffer && size != buffer->getInfo<CL_MEM_SIZE>()) {
                            delete buffer;
                            buffer = new cl::Buffer(mContext, flags, size);
                        } else if(!buffer) {
                            buffer = new cl::Buffer(mContext, flags, size);
                        }
                        return buffer;
                    }

                    cl::Buffer& get_device_grid(
                        unsigned int grid_size = 0)
                    {
                        if (grid_size > 0) {
                            auto size = auxiliary::sizeof_grid(grid_size);
                            d_grid = reuse_memory(size, d_grid, CL_MEM_READ_WRITE);
                        }
                        return *d_grid;
                    }

                    cl::Buffer& get_device_wavenumbers(
                        unsigned int nr_channels = 0)
                    {
                        if (nr_channels > 0)
                        {
                            auto size = auxiliary::sizeof_wavenumbers(nr_channels);
                            d_wavenumbers = reuse_memory(size, d_wavenumbers, CL_MEM_READ_WRITE);
                        }
                        return *d_wavenumbers;
                    }

                    cl::Buffer& get_device_aterms(
                        unsigned int nr_stations  = 0,
                        unsigned int nr_timeslots = 0,
                        unsigned int subgrid_size = 0)
                    {
                        if (nr_stations > 0 &&
                            nr_timeslots > 0 &&
                            subgrid_size > 0)
                        {
                            auto size = auxiliary::sizeof_aterms(nr_stations, nr_timeslots, subgrid_size);
                            d_aterms = reuse_memory(size, d_aterms, CL_MEM_READ_WRITE);
                        }
                        return *d_aterms;
                    }

                    cl::Buffer& get_device_spheroidal(
                        unsigned int subgrid_size = 0)
                    {
                        if (subgrid_size > 0)
                        {
                            auto size = auxiliary::sizeof_spheroidal(subgrid_size);
                            d_spheroidal = reuse_memory(size, d_spheroidal, CL_MEM_READ_WRITE);
                        }
                        return *d_spheroidal;
                    }

                    cl::Buffer& get_host_grid(
                        unsigned int grid_size)
                    {
                        if (grid_size > 0) {
                            auto size = auxiliary::sizeof_grid(grid_size);
                            h_grid = reuse_memory(size, h_grid, CL_MEM_ALLOC_HOST_PTR);
                        }
                        return *h_grid;
                    }

                    cl::Buffer& get_host_visibilities(
                        unsigned int nr_baselines,
                        unsigned int nr_timesteps,
                        unsigned int nr_channels)
                    {
                        if (nr_baselines > 0 &&
                            nr_timesteps > 0 &&
                            nr_channels > 0)
                        {
                            auto size = auxiliary::sizeof_visibilities(nr_baselines, nr_timesteps, nr_channels);
                            h_visibilities = reuse_memory(size, h_visibilities, CL_MEM_ALLOC_HOST_PTR);
                        }
                        return *h_visibilities;
                    }

                    cl::Buffer& get_host_uvw(
                        unsigned int nr_baselines,
                        unsigned int nr_timesteps)
                    {
                        if (nr_baselines > 0 &&
                            nr_timesteps > 0)
                        {
                            auto size = auxiliary::sizeof_uvw(nr_baselines, nr_timesteps);
                            h_uvw = reuse_memory(size, h_uvw, CL_MEM_ALLOC_HOST_PTR);
                        }
                        return *h_uvw;
                    }

                private:
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
