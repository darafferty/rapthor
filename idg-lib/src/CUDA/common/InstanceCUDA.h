#ifndef IDG_CUDA_INSTANCE_H_
#define IDG_CUDA_INSTANCE_H_

#include "idg-common.h"

#include "CU.h"
#include "CUFFT.h"
#include "PowerRecord.h"

namespace idg {
    namespace kernel {
        namespace cuda {

            class InstanceCUDA : public KernelsInstance {
                public:
                    // Constructor
                    InstanceCUDA(
                        CompileConstants &constants,
                        ProxyInfo &info,
                        int device_nr = 0,
                        int device_id = 0);

                    // Destructor
                    ~InstanceCUDA();

                    cu::Context& get_context() const { return *context; }
                    cu::Device&  get_device()  const { return *device; }
                    cu::Stream&  get_execute_stream() const { return *executestream; };
                    cu::Stream&  get_htod_stream() const { return *htodstream; };
                    cu::Stream&  get_dtoh_stream() const { return *dtohstream; };

                    void set_context() const {
                        context->setCurrent();
                    }

                    std::string get_compiler_flags();

                    powersensor::PowerSensor* get_powersensor() { return powerSensor; };
                    powersensor::State measure();
                    void measure(PowerRecord &record, cu::Stream &stream);

                    void launch_gridder(
                        int nr_subgrids,
                        int grid_size,
                        int subgrid_size,
                        float image_size,
                        float w_step,
                        int nr_channels,
                        int nr_stations,
                        cu::DeviceMemory& d_uvw,
                        cu::DeviceMemory& d_wavenumbers,
                        cu::DeviceMemory& d_visibilities,
                        cu::DeviceMemory& d_spheroidal,
                        cu::DeviceMemory& d_aterm,
                        cu::DeviceMemory& d_metadata,
                        cu::DeviceMemory& d_subgrid);

                    void launch_degridder(
                        int nr_subgrids,
                        int grid_size,
                        int subgrid_size,
                        float image_size,
                        float w_step,
                        int nr_channels,
                        int nr_stations,
                        cu::DeviceMemory& d_uvw,
                        cu::DeviceMemory& d_wavenumbers,
                        cu::DeviceMemory& d_visibilities,
                        cu::DeviceMemory& d_spheroidal,
                        cu::DeviceMemory& d_aterm,
                        cu::DeviceMemory& d_metadata,
                        cu::DeviceMemory& d_subgrid);

                    void plan_fft(
                        int size, int batch);

                    void launch_fft(
                        cu::DeviceMemory& d_data,
                        DomainAtoDomainB direction);

                    void launch_adder(
                        int nr_subgrids,
                        int grid_size,
                        int subgrid_size,
                        cu::DeviceMemory& d_metadata,
                        cu::DeviceMemory& d_subgrid,
                        cu::DeviceMemory& d_grid);

                    void launch_splitter(
                        int nr_subgrids,
                        int grid_size,
                        int subgrid_size,
                        cu::DeviceMemory& d_metadata,
                        cu::DeviceMemory& d_subgrid,
                        cu::DeviceMemory& d_grid);

                    void launch_scaler(
                        int nr_subgrids,
                        int subgrid_size,
                        cu::DeviceMemory& d_subgrid);


                    cu::HostMemory& get_host_grid(
                        unsigned int grid_size);

                    cu::DeviceMemory& get_device_grid(
                        unsigned int grid_size);

                    cu::DeviceMemory& get_device_wavenumbers(
                        unsigned int nr_channels);

                    cu::DeviceMemory& get_device_aterms(
                        unsigned int nr_stations,
                        unsigned int nr_timeslots,
                        unsigned int subgrid_size);

                    cu::DeviceMemory& get_device_spheroidal(
                        unsigned int subgrid_size);

                    cu::HostMemory& get_host_grid(
                        unsigned int grid_size,
                        void *ptr);

                    cu::HostMemory& get_host_visibilities(
                        unsigned int nr_baselines,
                        unsigned int nr_timesteps,
                        unsigned int nr_channels,
                        void *ptr);

                    cu::HostMemory& get_host_uvw(
                        unsigned int nr_baselines,
                        unsigned int nr_timesteps,
                        void *ptr);

                    cu::HostMemory& get_host_grid() { return *h_grid; }

                    cu::DeviceMemory& get_device_grid() { return *d_grid; }

                    cu::DeviceMemory& get_device_wavenumbers() { return *d_wavenumbers; }

                    cu::DeviceMemory& get_device_aterms() { return *d_aterms; }

                    cu::DeviceMemory& get_device_spheroidal() { return *d_spheroidal; }

                protected:
                    void compile_kernels();
                    void load_kernels();
                    void set_parameters();
                    void set_parameters_kepler();
                    void set_parameters_maxwell();
                    void set_parameters_pascal();

                protected:
                    // Variables shared by all InstanceCUDA instances
                    ProxyInfo &mInfo;

                private:
                    cu::Context *context;
                    cu::Device  *device;
                    cu::Stream  *executestream;
                    cu::Stream  *htodstream;
                    cu::Stream  *dtohstream;
                    cu::Function *function_gridder;
                    cu::Function *function_degridder;
                    cu::Function *function_fft;
                    cu::Function *function_adder;
                    cu::Function *function_splitter;
                    cu::Function *function_scaler;
                    cu::HostMemory *h_visibilities;
                    cu::HostMemory *h_uvw;
                    cu::HostMemory *h_grid;
                    cu::DeviceMemory *d_grid;
                    cu::DeviceMemory *d_wavenumbers;
                    cu::DeviceMemory *d_aterms;
                    cu::DeviceMemory *d_spheroidal;
                    std::vector<cu::HostMemory*> h_visibilities_;
                    std::vector<cu::HostMemory*> h_uvw_;
                    std::vector<cu::HostMemory*> h_grid_;

                    // All CUDA modules private to this InstanceCUDA
                    std::vector<cu::Module*> mModules;

                    // Power sensor private to this InstanceCUDA
                    powersensor::PowerSensor *powerSensor;

                protected:
                    dim3 block_gridder;
                    dim3 block_degridder;
                    dim3 block_adder;
                    dim3 block_splitter;
                    dim3 block_scaler;


                    // (De)gridder kernel
                    int batch_gridder;
                    int batch_degridder;

                    // FFT kernel
                    const int fft_bulk = 1024;
                    int fft_batch;
                    int fft_size;
                    cufft::C2C_2D *fft_plan_bulk;
                    cufft::C2C_2D *fft_plan_misc;

            };

            std::ostream& operator<<(std::ostream& os, InstanceCUDA &d);

            // Kernel names
            static const std::string name_gridder   = "kernel_gridder";
            static const std::string name_degridder = "kernel_degridder";
            static const std::string name_adder     = "kernel_adder";
            static const std::string name_splitter  = "kernel_splitter";
            static const std::string name_fft       = "kernel_fft";
            static const std::string name_scaler    = "kernel_scaler";

        } // end namespace cuda
    } // end namespace kernel
} // end namespace idg

#endif
