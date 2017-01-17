#ifndef IDG_CUDA_DEVICEINSTANCE_H_
#define IDG_CUDA_DEVICEINSTANCE_H_

#include <memory> // unique_ptr

#include "idg-common.h"

#include "CU.h"
#include "CUFFT.h"
#include "PowerRecord.h"

namespace idg {
    namespace kernel {
        namespace cuda {

            // Kernel names
            static const std::string name_gridder   = "kernel_gridder";
            static const std::string name_degridder = "kernel_degridder";
            static const std::string name_adder     = "kernel_adder";
            static const std::string name_splitter  = "kernel_splitter";
            static const std::string name_fft       = "kernel_fft";
            static const std::string name_scaler    = "kernel_scaler";

            /*
                Kernel classes
            */
            class Gridder {
                public:
                    Gridder(
                        cu::Module &module,
                        const dim3 block);

                    void launch(
                        cu::Stream &stream,
                        int nr_subgrids,
                        int gridsize,
                        float imagesize,
                        float w_offset,
                        int nr_channels,
                        int nr_stations,
                        cu::DeviceMemory &d_uvw,
                        cu::DeviceMemory &d_wavenumbers,
                        cu::DeviceMemory &d_visibilities,
                        cu::DeviceMemory &d_spheroidal,
                        cu::DeviceMemory &d_aterm,
                        cu::DeviceMemory &d_metadata,
                        cu::DeviceMemory &d_subgrid);

                private:
                    cu::Function function;
                    dim3 block;
            };


            class Degridder {
                public:
                    Degridder(
                        cu::Module &module,
                        const dim3 block);

                    void launch(
                        cu::Stream &stream,
                        int nr_subgrids,
                        int gridsize,
                        float imagesize,
                        float w_offset,
                        int nr_channels,
                        int nr_stations,
                        cu::DeviceMemory &d_uvw,
                        cu::DeviceMemory &d_wavenumbers,
                        cu::DeviceMemory &d_visibilities,
                        cu::DeviceMemory &d_spheroidal,
                        cu::DeviceMemory &d_aterm,
                        cu::DeviceMemory &d_metadata,
                        cu::DeviceMemory &d_subgrid);

                private:
                    cu::Function function;
                    dim3 block;
            };


            class GridFFT {
                public:
                    GridFFT(
                        unsigned int nr_correlations,
                        unsigned int size,
                        cu::Module &module);

                    ~GridFFT();

                    void plan(
                        unsigned int batch);

                    void launch(cu::Stream &stream, cu::DeviceMemory &data, int direction);

                private:
                    cu::Function function;
                    unsigned int nr_correlations;
                    unsigned int size;
                    unsigned int planned_batch;
                    const unsigned int bulk_size = 1024;
                    cufft::C2C_2D *fft_bulk;
                    cufft::C2C_2D *fft_remainder;
            };


            class Adder {
                public:
                    Adder(
                        cu::Module &module,
                        const dim3 block);

                    void launch(
                        cu::Stream &stream,
                        int nr_subgrids,
                        int gridsize,
                        cu::DeviceMemory &d_metadata,
                        cu::DeviceMemory &d_subgrid,
                        cu::DeviceMemory &d_grid);

                private:
                    cu::Function function;
                    dim3 block;
            };


            class Splitter {
                public:
                    Splitter(
                        cu::Module &module,
                        const dim3 block);

                    void launch(
                        cu::Stream &stream,
                        int nr_subgrids,
                        int gridsize,
                        cu::DeviceMemory &d_metadata,
                        cu::DeviceMemory &d_subgrid,
                        cu::DeviceMemory &d_grid);

                private:
                    cu::Function function;
                    dim3 block;
            };


            class Scaler {
                public:
                    Scaler(
                        cu::Module &module,
                        const dim3 block);

                    void launch(
                        cu::Stream &stream,
                        int nr_subgrids,
                        cu::DeviceMemory &d_subgrid);

                private:
                    cu::Function function;
                    dim3 block;
            };


            /*
                DeviceInstance
            */
            class DeviceInstance : public Kernels {
                public:
                    // Constructor
                    DeviceInstance(
                        CompileConstants &constants,
                        ProxyInfo &info,
                        int device_number,
                        const char *power_sensor = NULL,
                        const char *power_file = NULL);

                    // Destructor
                    ~DeviceInstance();

                    std::unique_ptr<Gridder> get_kernel_gridder() const {
                        return std::unique_ptr<Gridder>(new Gridder(
                            *(modules[which_module.at(name_gridder)]), block_gridder));
                    }

                    std::unique_ptr<Degridder> get_kernel_degridder() const {
                        return std::unique_ptr<Degridder>(new Degridder(
                            *(modules[which_module.at(name_degridder)]), block_degridder));
                    }

                    std::unique_ptr<GridFFT> get_kernel_fft(unsigned int size) const {
                        return std::unique_ptr<GridFFT>(new GridFFT(
                            mConstants.get_nr_correlations(),
                            size,
                            *(modules[which_module.at(name_fft)])));
                    }

                    std::unique_ptr<Adder> get_kernel_adder() const {
                        return std::unique_ptr<Adder>(new Adder(
                            *(modules[which_module.at(name_adder)]), block_adder));
                    }

                    std::unique_ptr<Splitter> get_kernel_splitter() const {
                        return std::unique_ptr<Splitter>(new Splitter(
                            *(modules[which_module.at(name_splitter)]), block_splitter));
                    }

                    std::unique_ptr<Scaler> get_kernel_scaler() const {
                        return std::unique_ptr<Scaler>(new Scaler(
                            *(modules[which_module.at(name_scaler)]), block_scaler));
                    }

                    cu::Context& get_context() const { return *context; }
                    cu::Device&  get_device()  const { return *device; }
                    cu::Stream&  get_execute_stream() const { return *executestream; };
                    cu::Stream&  get_htod_stream() const { return *htodstream; };
                    cu::Stream&  get_dtoh_stream() const { return *dtohstream; };

                    void set_context() const {
                        context->setCurrent();
                    }

                    std::string get_compiler_flags();

                    PowerSensor* get_powersensor() { return powerSensor; };
                    PowerSensor::State measure();
                    void measure(
                        idg::kernel::cuda::PowerRecord &record, cu::Stream &stream);

                    cu::HostMemory& allocate_host_grid(
                        unsigned int grid_size);

                    cu::DeviceMemory& allocate_device_grid(
                        unsigned int grid_size);

                    cu::HostMemory& allocate_host_visibilities(
                        unsigned int nr_baselines,
                        unsigned int nr_timesteps,
                        unsigned int nr_channels);

                    cu::HostMemory& allocate_host_uvw(
                        unsigned int nr_baselines,
                        unsigned int nr_timesteps);

                    cu::DeviceMemory& allocate_device_wavenumbers(
                        unsigned int nr_channels);

                    cu::DeviceMemory& allocate_device_aterms(
                        unsigned int nr_stations,
                        unsigned int nr_timeslots,
                        unsigned int subgrid_size);

                    cu::DeviceMemory& allocate_device_spheroidal(
                        unsigned int subgrid_size);

                    cu::HostMemory reuse_host_grid(
                        unsigned int grid_size,
                        void *ptr);

                    cu::HostMemory& get_host_grid() { return *h_grid; }

                    cu::DeviceMemory& get_device_grid() { return *d_grid; }

                    cu::HostMemory& get_host_visibilities() { return *h_visibilities; }

                    cu::HostMemory& get_host_uvw() { return *h_uvw; }

                    cu::DeviceMemory& get_device_wavenumbers() { return *d_wavenumbers; }

                    cu::DeviceMemory& get_device_aterms() { return *d_aterms; }

                    cu::DeviceMemory& get_device_spheroidal() { return *d_spheroidal; }

                protected:
                    void compile_kernels();
                    void load_modules();
                    void set_parameters();
                    void set_parameters_kepler();
                    void set_parameters_maxwell();
                    void set_parameters_pascal();
                    void init_powersensor(
                        const char *str_power_sensor,
                        const char *str_power_file);

                protected:
                    // Variables shared by all DeviceInstance instances
                    ProxyInfo &mInfo;

                private:
                    // CUDA objects private to this DeviceInstance
                    cu::Context *context;
                    cu::Device  *device;
                    cu::Stream  *executestream;
                    cu::Stream  *htodstream;
                    cu::Stream  *dtohstream;
                    cu::HostMemory *h_visibilities;
                    cu::HostMemory *h_uvw;
                    cu::HostMemory *h_grid;
                    cu::DeviceMemory *d_grid;
                    cu::DeviceMemory *d_wavenumbers;
                    cu::DeviceMemory *d_aterms;
                    cu::DeviceMemory *d_spheroidal;

                    // All CUDA modules private to this DeviceInstance
                    std::vector<cu::Module*> modules;
                    std::map<std::string,int> which_module;

                    // Power sensor private to this DeviceInstance
                    PowerSensor *powerSensor;

                protected:
                    dim3 block_gridder;
                    dim3 block_degridder;
                    dim3 block_adder;
                    dim3 block_splitter;
                    dim3 block_scaler;
                    int batch_gridder;
                    int batch_degridder;
            };

            std::ostream& operator<<(std::ostream& os, DeviceInstance &d);
        } // end namespace cuda
    } // end namespace kernel
} // end namespace idg

#endif
