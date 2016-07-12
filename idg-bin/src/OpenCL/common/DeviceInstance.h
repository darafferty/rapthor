#ifndef IDG_OPENCL_DEVICEINSTANCE_H_
#define IDG_OPENCL_DEVICEINSTANCE_H_

#include <cstring>
#include <sstream>
#include <memory>

#include "idg-common.h"
#include "idg-powersensor.h"

#include "Kernels.h"
#include "Util.h"

namespace idg {
    namespace proxy {
        namespace opencl {
            class DeviceInstance {
                public:
                    DeviceInstance(
                        Parameters &params,
                        ProxyInfo &info,
                        int device_number,
                        const char *power_sensor = NULL,
                        const char *power_file = NULL);

                    cl::Context& get_context() const { return *context; }
                    cl::Device&  get_device()  const { return *device; }
                    cl::CommandQueue&  get_execute_queue() const { return *executequeue; };
                    cl::CommandQueue&  get_htod_queue() const { return *htodqueue; };
                    cl::CommandQueue&  get_dtoh_queue() const { return *dtohqueue; };

                    std::unique_ptr<kernel::opencl::Gridder>   get_kernel_gridder() const;
                    std::unique_ptr<kernel::opencl::Degridder> get_kernel_degridder() const;
                    std::unique_ptr<kernel::opencl::GridFFT>   get_kernel_fft() const;
                    std::unique_ptr<kernel::opencl::Adder>     get_kernel_adder() const;
                    std::unique_ptr<kernel::opencl::Splitter>  get_kernel_splitter() const;
                    std::unique_ptr<kernel::opencl::Scaler>    get_kernel_scaler() const;

                    std::string get_compiler_flags();

                    PowerSensor* get_powersensor() { return powerSensor; };

                protected:
                    void compile_kernels();
                    void load_modules();
                    void set_parameters();
                    void set_parameters_fiji();
                    void set_parameters_hawaii();
                    void set_parameters_tahiti();
                    void init_powersensor(
                        const char *str_power_sensor,
                        const char *str_power_file);

                protected:
                    // Arguments shared by all DeviceInstance instances
                    Parameters  &parameters;
                    ProxyInfo   &info;

                private:
                    // OpenCL objects private to this DeviceInstance
                    cl::Context *context;
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
    } // end namespace proxy
} // end namespace idg

#endif
