#ifndef IDG_OPENCL_DEVICEINSTANCE_H_
#define IDG_OPENCL_DEVICEINSTANCE_H_

#include <cstring>
#include <sstream>
#include <memory>

#include "idg-common.h"
#include "idg-powersensor.h"

#include "Util.h"

namespace idg {
    namespace kernel {
        namespace opencl {
            class PerformanceCounter;
            typedef size_t clfftPlanHandle;

			// Kernel names
			static const std::string name_gridder   = "kernel_gridder";
			static const std::string name_degridder = "kernel_degridder";
			static const std::string name_adder     = "kernel_adder";
			static const std::string name_splitter  = "kernel_splitter";
			static const std::string name_scaler    = "kernel_scaler";

            /*
                Kernel classes
            */
            class Gridder {
                public:
                    Gridder(
						cl::Program &program,
						const cl::NDRange &local_size);
                     void launchAsync(
                        cl::CommandQueue &queue,
                        int nr_baselines,
                        int nr_subgrids,
                        int gridsize,
                        float imagesize,
                        float w_offset,
                        int nr_channels,
                        int nr_stations,
                        cl::Buffer &d_uvw,
                        cl::Buffer &d_wavenumbers,
                        cl::Buffer &d_visibilities,
                        cl::Buffer &d_spheroidal,
                        cl::Buffer &d_aterm,
                        cl::Buffer &d_metadata,
                        cl::Buffer &d_subgrid,
                        PerformanceCounter &counter);

                private:
                    cl::Event event;
                    cl::Kernel kernel;
					cl::NDRange local_size;
            };


            class Degridder {
                public:
                    Degridder(
						cl::Program &program,
						const cl::NDRange &local_size);
                    void launchAsync(
                        cl::CommandQueue &queue,
                        int nr_baselines,
                        int nr_subgrids,
                        int gridsize,
                        float imagesize,
                        float w_offset,
                        int nr_channels,
                        int nr_stations,
                        cl::Buffer &d_uvw,
                        cl::Buffer &d_wavenumbers,
                        cl::Buffer &d_visibilities,
                        cl::Buffer &d_spheroidal,
                        cl::Buffer &d_aterm,
                        cl::Buffer &d_metadata,
                        cl::Buffer &d_subgrid,
                        PerformanceCounter &counter);

                private:
                    cl::Event event;
                    cl::Kernel kernel;
					cl::NDRange local_size;
            };


            class GridFFT {
                public:
                    GridFFT(
                        unsigned int nr_correlations);
                    ~GridFFT();
                    void plan(
                        cl::Context &context, cl::CommandQueue &queue,
                        int size, int batch);
                    void launchAsync(
                        cl::CommandQueue &queue,
                        cl::Buffer &d_data,
                        DomainAtoDomainB direction);
                    void launchAsync(
                        cl::CommandQueue &queue,
                        cl::Buffer &d_data,
                        DomainAtoDomainB direction,
                        PerformanceCounter &counter,
                        const char *name);
                    void shift(std::complex<float> *data);
                    void scale(std::complex<float> *data, std::complex<float> scale);

                private:
                    bool uninitialized;
                    unsigned int nr_correlations;
                    int planned_size;
                    int planned_batch;
                    clfftPlanHandle fft;
                    cl::Event start;
                    cl::Event end;
            };

            class Adder {
                public:
                    Adder(
						cl::Program &program,
						const cl::NDRange &local_size);
                    void launchAsync(
                        cl::CommandQueue &queue,
                        int nr_subgrids,
                        int gridsize,
                        cl::Buffer d_metadata,
                        cl::Buffer d_subgrid,
                        cl::Buffer d_grid,
                        PerformanceCounter &counter);

                private:
                    cl::Event event;
                    cl::Kernel kernel;
					cl::NDRange local_size;
            };

            class Splitter {
                public:
                    Splitter(
						cl::Program &program,
						const cl::NDRange &local_size);
                    void launchAsync(
                        cl::CommandQueue &queue,
                        int nr_subgrids,
                        int gridsize,
                        cl::Buffer d_metadata,
                        cl::Buffer d_subgrid,
                        cl::Buffer d_grid,
                        PerformanceCounter &counter);

                private:
                    cl::Event event;
                    cl::Kernel kernel;
					cl::NDRange local_size;
            };

            class Scaler {
                public:
                    Scaler(
						cl::Program &program,
						const cl::NDRange &local_size);
                    void launchAsync(
                        cl::CommandQueue &queue,
                        int nr_subgrids,
                        cl::Buffer d_subgrid,
                        PerformanceCounter &counter);

                private:
                    cl::Event event;
                    cl::Kernel kernel;
					cl::NDRange local_size;
            };



            class DeviceInstance : public Kernels {
                public:
                    DeviceInstance(
                        CompileConstants &constants,
                        cl::Context &context,
                        int device_number,
                        const char *power_sensor = NULL,
                        const char *power_file = NULL);

                    ~DeviceInstance();

                    cl::Device&  get_device()  const { return *device; }
                    cl::CommandQueue&  get_execute_queue() const { return *executequeue; };
                    cl::CommandQueue&  get_htod_queue() const { return *htodqueue; };
                    cl::CommandQueue&  get_dtoh_queue() const { return *dtohqueue; };

                    std::unique_ptr<Gridder>   get_kernel_gridder() const;
                    std::unique_ptr<Degridder> get_kernel_degridder() const;
                    std::unique_ptr<GridFFT>   get_kernel_fft() const;
                    std::unique_ptr<Adder>     get_kernel_adder() const;
                    std::unique_ptr<Splitter>  get_kernel_splitter() const;
                    std::unique_ptr<Scaler>    get_kernel_scaler() const;

                    std::string get_compiler_flags();

                    PowerSensor* get_powersensor() { return powerSensor; };

                protected:
                    void compile_kernels(cl::Context &context);
                    void load_modules();
                    void set_parameters();
                    void set_parameters_default();
                    void set_parameters_fiji();
                    void set_parameters_hawaii();
                    void set_parameters_tahiti();
                    void init_powersensor(
                        const char *str_power_sensor,
                        const char *str_power_file);

                private:
                    // OpenCL objects private to this DeviceInstance
                    cl::Device *device;
                    cl::CommandQueue  *executequeue;
                    cl::CommandQueue  *htodqueue;
                    cl::CommandQueue  *dtohqueue;

                    // All OpenCL programs private to this DeviceInstance
                    std::vector<cl::Program*> programs;
                    std::map<std::string,int> which_program;

                    // Power sensor private to this DeviceInstance
                    PowerSensor *powerSensor;

                protected:
                    cl::NDRange block_gridder;
                    cl::NDRange block_degridder;
                    cl::NDRange block_adder;
                    cl::NDRange block_splitter;
                    cl::NDRange block_scaler;
                    int batch_gridder;
                    int batch_degridder;
            };

            std::ostream& operator<<(std::ostream& os, DeviceInstance &d);
        } // end namespace opencl
    } // end namespace kernel
} // end namespace idg

#endif
