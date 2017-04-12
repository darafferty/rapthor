#ifndef IDG_OPENCL_INSTANCE_H_
#define IDG_OPENCL_INSTANCE_H_

#include <cstring>
#include <sstream>
#include <memory>

#include "idg-common.h"
#include "idg-powersensor.h"

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
                        int device_number,
                        const char *power_sensor = NULL,
                        const char *power_file = NULL);

                    // Destructor
                    ~InstanceOpenCL();

                    cl::Device&       get_device()  const { return *device; }
                    cl::CommandQueue& get_execute_queue() const { return *executequeue; };
                    cl::CommandQueue& get_htod_queue() const { return *htodqueue; };
                    cl::CommandQueue& get_dtoh_queue() const { return *dtohqueue; };

                    std::string get_compiler_flags();

                    PowerSensor* get_powersensor() { return powerSensor; };
                    PowerSensor::State measure();
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
                    void compile_kernels();
                    void load_kernels();
                    void set_parameters();
                    void set_parameters_default();
                    void set_parameters_fiji();
                    void set_parameters_hawaii();
                    void set_parameters_tahiti();
                    void init_powersensor(
                        const char *str_power_sensor,
                        const char *str_power_file);

                private:
                    cl::Context& mContext;
                    cl::Device *device;
                    cl::CommandQueue  *executequeue;
                    cl::CommandQueue  *htodqueue;
                    cl::CommandQueue  *dtohqueue;
                    cl::Kernel *kernel_gridder;
                    cl::Kernel *kernel_degridder;
                    cl::Kernel *kernel_fft;
                    cl::Kernel *kernel_adder;
                    cl::Kernel *kernel_splitter;
                    cl::Kernel *kernel_scaler;
                    std::vector<cl::Program*> mPrograms;
                    PowerSensor *powerSensor;

                protected:
                    cl::NDRange block_gridder;
                    cl::NDRange block_degridder;
                    cl::NDRange block_adder;
                    cl::NDRange block_splitter;
                    cl::NDRange block_scaler;

                    // (De)gridder kernel
                    int batch_gridder;
                    int batch_degridder;

                    // FFT kernel
                    bool fft_planned;
                    unsigned int fft_planned_size;
                    unsigned int fft_planned_batch;
                    clfftPlanHandle fft_plan;

            };

            std::ostream& operator<<(std::ostream& os, InstanceOpenCL &d);

			// Kernel names
			static const std::string name_gridder   = "kernel_gridder";
			static const std::string name_degridder = "kernel_degridder";
			static const std::string name_adder     = "kernel_adder";
			static const std::string name_splitter  = "kernel_splitter";
			static const std::string name_scaler    = "kernel_scaler";

        } // end namespace opencl
    } // end namespace kernel
} // end namespace idg

#endif
