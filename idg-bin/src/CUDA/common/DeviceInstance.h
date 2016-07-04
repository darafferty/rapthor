#ifndef IDG_CUDA_DEVICEINSTANCE_H_
#define IDG_CUDA_DEVICEINSTANCE_H_

#include <cstring>
#include <sstream>
#include <memory>

#include "CU.h"
#include "Kernels.h"
#include "PowerRecord.h"

#include "common/Parameters.h"
#include "common/ProxyInfo.h"

namespace idg {
    namespace proxy {
        namespace cuda {
            /*
                Power measurement
            */
            class PowerRecord {
                public:
                    static void getPower(CUstream, CUresult, void *userData);
                    PowerSensor *sensor;
                    PowerSensor::State state;
                    cu::Event event;
            };

            class DeviceInstance {
                public:
                    DeviceInstance(
                        Parameters &params,
                        ProxyInfo &info,
                        int device_number);

                    cu::Context& get_context() const { return *context; }
                    cu::Device&  get_device()  const { return *device; }

                    std::unique_ptr<kernel::cuda::Gridder>   get_kernel_gridder() const;
                    std::unique_ptr<kernel::cuda::Degridder> get_kernel_degridder() const;
                    std::unique_ptr<kernel::cuda::GridFFT>   get_kernel_fft() const;
                    std::unique_ptr<kernel::cuda::Adder>     get_kernel_adder() const;
                    std::unique_ptr<kernel::cuda::Splitter>  get_kernel_splitter() const;
                    std::unique_ptr<kernel::cuda::Scaler>    get_kernel_scaler() const;

                    std::string get_compiler_flags();

                    PowerSensor::State measure();
                    void measure(PowerRecord &record, cu::Stream &stream);

                protected:
                    void compile_kernels();
                    void load_modules();
                    void set_parameters();
                    void set_parameters_kepler();
                    void set_parameters_maxwell();
                    void set_parameters_pascal();
                    void init_powersensor();

                protected:
                    // Arguments shared by all DeviceInstance instances
                    Parameters  &parameters;
                    ProxyInfo   &info;

                private:
                    // CUDA objects private to this DeviceInstance
                    cu::Context *context;
                    cu::Device  *device;

                    // All CUDA modules private to this DeviceInstance
                    std::vector<cu::Module*> modules;
                    std::map<std::string,int> which_module;

                    PowerSensor powerSensor;

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
        }
    }
}

#endif
