#include <cstdint> // unint64_t
#include <unistd.h> // rmdir

#include <algorithm> // transform

#include "idg-config.h"

#include "InstanceCPU.h"

using namespace std;

namespace idg {
    namespace kernel {
        namespace cpu {

            // Constructor
            InstanceCPU::InstanceCPU(
                std::vector<std::string> libraries):
                KernelsInstance(),
                function_gridder(nullptr),
                function_degridder(nullptr),
                function_fft(nullptr),
                function_adder(nullptr),
                function_splitter(nullptr),
                function_adder_wstack(nullptr),
                function_splitter_wstack(nullptr)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                powerSensor = powersensor::get_power_sensor(powersensor::sensor_host);

                load_shared_objects(libraries);
                load_kernel_funcions();
            }

            // Destructor
            InstanceCPU::~InstanceCPU()
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Unload shared objects by ~Module
                for (unsigned int i = 0; i < modules.size(); i++) {
                    delete modules[i];
                }

                // Unload functions
                delete function_gridder;
                delete function_degridder;
                delete function_fft;
                delete function_adder;
                delete function_splitter;
                delete function_adder_wstack;
                delete function_splitter_wstack;

                // Delete power sensor
                delete powerSensor;
            }

            void InstanceCPU::load_shared_objects(
                std::vector<std::string> libraries)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                for (auto library : libraries) {
                    string library_ = auxiliary::get_lib_dir() + "/idg-cpu/" + library;

                    #if defined(DEBUG)
                    cout << "Loading: " << library_ << endl;
                    #endif

                    modules.push_back(new runtime::Module(library_.c_str()));
                }
            } // end load_shared_objects

            // maps name -> index in modules that contain that symbol
            void InstanceCPU::load_kernel_funcions()
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                for (unsigned int i=0; i<modules.size(); i++) {
                    if (dlsym(*modules[i], kernel::cpu::name_gridder.c_str())) {
                        function_gridder = new runtime::Function(*modules[i], name_gridder.c_str());
                    }
                    if (dlsym(*modules[i], kernel::cpu::name_degridder.c_str())) {
                        function_degridder = new runtime::Function(*modules[i], name_degridder.c_str());
                    }
                    if (dlsym(*modules[i], kernel::cpu::name_fft.c_str())) {
                        function_fft = new runtime::Function(*modules[i], name_fft.c_str());
                    }
                    if (dlsym(*modules[i], kernel::cpu::name_adder.c_str())) {
                        function_adder = new runtime::Function(*modules[i], name_adder.c_str());
                    }
                    if (dlsym(*modules[i], kernel::cpu::name_splitter.c_str())) {
                        function_splitter = new runtime::Function(*modules[i], name_splitter.c_str());
                    }
                    if (dlsym(*modules[i], kernel::cpu::name_adder_wstack.c_str())) {
                        function_adder_wstack = new runtime::Function(*modules[i], name_adder_wstack.c_str());
                    }
                    if (dlsym(*modules[i], kernel::cpu::name_splitter_wstack.c_str())) {
                        function_splitter_wstack = new runtime::Function(*modules[i], name_splitter_wstack.c_str());
                    }
                } // end for
            } // end load_kernel_funcions

            // Function signatures
            #define sig_gridder         (void (*)(int,int,int,float,float,const float*,int,int,void*,void*,void*,void*,void*,void*,void*,void*))
            #define sig_degridder       (void (*)(int,int,int,float,float,const float*,int,int,void*,void*,void*,void*,void*,void*,void*))
            #define sig_fft		        (void (*)(long,long,long,void*,int))
            #define sig_adder	        (void (*)(long,long,int,void*,void*,void*))
            #define sig_splitter        (void (*)(long,long,int,void*,void*,void*))
            #define sig_adder_wstack    (void (*)(long,long,int,void*,void*,void*))
            #define sig_splitter_wstack (void (*)(long,long,int,void*,void*,void*))


            void InstanceCPU::run_gridder(
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
                void *subgrid)
            {
                powersensor::State states[2];
                states[0] = powerSensor->read();
                (sig_gridder (void *) *function_gridder)(
                  nr_subgrids, grid_size, subgrid_size, image_size, w_step, shift, nr_channels, nr_stations,
                  uvw, wavenumbers, visibilities, spheroidal, aterm, avg_aterm, metadata, subgrid);
                states[1] = powerSensor->read();
                if (report) { report->update_gridder(states[0], states[1]); }
            }

            void InstanceCPU::run_degridder(
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
                void *subgrid)
            {
                powersensor::State states[2];
                states[0] = powerSensor->read();
                (sig_degridder (void *) *function_degridder)(
                  nr_subgrids, grid_size, subgrid_size, image_size, w_step, shift, nr_channels, nr_stations,
                  uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrid);
                states[1] = powerSensor->read();
                if (report) { report->update_degridder(states[0], states[1]); }
            }

            void InstanceCPU::run_fft(
                int grid_size,
                int size,
                int batch,
                void *data,
                int direction)
            {
                powersensor::State states[2];
                states[0] = powerSensor->read();
                (sig_fft (void *) *function_fft)(grid_size, size, batch, data, direction);
                states[1] = powerSensor->read();
                if (report) { report->update_grid_fft(states[0], states[1]); }
            }

             void InstanceCPU::run_subgrid_fft(
                int grid_size,
                int size,
                int batch,
                void *data,
                int direction)
            {
                powersensor::State states[2];
                states[0] = powerSensor->read();
                (sig_fft (void *) *function_fft)(grid_size, size, batch, data, direction);
                states[1] = powerSensor->read();
                if (report) { report->update_subgrid_fft(states[0], states[1]); }
            }

            void InstanceCPU::run_adder(
                int nr_subgrids,
                int grid_size,
                int subgrid_size,
                void *metadata,
                void *subgrid,
                void *grid)
            {
                powersensor::State states[2];
                states[0] = powerSensor->read();
                (sig_adder (void *) *function_adder)(nr_subgrids, grid_size, subgrid_size, metadata, subgrid, grid);
                states[1] = powerSensor->read();
                if (report) { report->update_adder(states[0], states[1]); }
            }

            void InstanceCPU::run_splitter(
                int nr_subgrids,
                int grid_size,
                int subgrid_size,
                void *metadata,
                void *subgrid,
                void *grid)
            {
                powersensor::State states[2];
                states[0] = powerSensor->read();
                (sig_splitter (void *) *function_splitter)(nr_subgrids, grid_size, subgrid_size, metadata, subgrid, grid);
                states[1] = powerSensor->read();
                if (report) { report->update_splitter(states[0], states[1]); }
            }

            void InstanceCPU::run_adder_wstack(
                int nr_subgrids,
                int grid_size,
                int subgrid_size,
                void *metadata,
                void *subgrid,
                void *grid)
            {
                powersensor::State states[2];
                states[0] = powerSensor->read();
                (sig_adder_wstack (void *) *function_adder_wstack)(nr_subgrids, grid_size, subgrid_size, metadata, subgrid, grid);
                states[1] = powerSensor->read();
                if (report) { report->update_adder(states[0], states[1]); }
            }

            void InstanceCPU::run_splitter_wstack(
                int nr_subgrids,
                int grid_size,
                int subgrid_size,
                void *metadata,
                void *subgrid,
                void *grid)
            {
                powersensor::State states[2];
                states[0] = powerSensor->read();
                (sig_splitter_wstack (void *) *function_splitter_wstack)(nr_subgrids, grid_size, subgrid_size, metadata, subgrid, grid);
                states[1] = powerSensor->read();
                if (report) { report->update_splitter(states[0], states[1]); }
            }

        } // namespace cpu
    } // namespace kernel
} // namespace idg
