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
                function_calibrate(nullptr),
                function_calibrate_hessian_vector_product1(nullptr),
                function_calibrate_hessian_vector_product2(nullptr),
                function_phasor(nullptr),
                function_fft(nullptr),
                function_adder(nullptr),
                function_splitter(nullptr),
                function_adder_wstack(nullptr),
                function_splitter_wstack(nullptr),
                function_adder_wtiles_to_grid(nullptr),
                function_splitter_wtiles_from_grid(nullptr),
                function_adder_subgrids_to_wtiles(nullptr),
                function_splitter_subgrids_from_wtiles(nullptr)
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
                delete function_calibrate;
                delete function_calibrate_hessian_vector_product1;
                delete function_calibrate_hessian_vector_product2;
                delete function_phasor;
                delete function_fft;
                delete function_adder;
                delete function_splitter;
                delete function_adder_wstack;
                delete function_splitter_wstack;
                delete function_adder_wtiles_to_grid;
                delete function_splitter_wtiles_from_grid;
                delete function_adder_subgrids_to_wtiles;
                delete function_splitter_subgrids_from_wtiles;

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
                    if (dlsym(*modules[i], kernel::cpu::name_calibrate.c_str())) {
                        function_calibrate = new runtime::Function(*modules[i], name_calibrate.c_str());
                    }
                    if (dlsym(*modules[i], kernel::cpu::name_calibrate_hessian_vector_product1.c_str())) {
                        function_calibrate_hessian_vector_product1 = new runtime::Function(*modules[i], name_calibrate_hessian_vector_product1.c_str());
                    }
                    if (dlsym(*modules[i], kernel::cpu::name_calibrate_hessian_vector_product1.c_str())) {
                        function_calibrate_hessian_vector_product2 = new runtime::Function(*modules[i], name_calibrate_hessian_vector_product2.c_str());
                    }
                    if (dlsym(*modules[i], kernel::cpu::name_phasor.c_str())) {
                        function_phasor = new runtime::Function(*modules[i], name_phasor.c_str());
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
                    if (dlsym(*modules[i], kernel::cpu::name_adder_wtiles_to_grid.c_str())) {
                        function_adder_wtiles_to_grid = new runtime::Function(*modules[i], name_adder_wtiles_to_grid.c_str());
                    }
                    if (dlsym(*modules[i], kernel::cpu::name_splitter_wtiles_from_grid.c_str())) {
                        function_splitter_wtiles_from_grid = new runtime::Function(*modules[i], name_splitter_wtiles_from_grid.c_str());
                    }
                    if (dlsym(*modules[i], kernel::cpu::name_adder_subgrids_to_wtiles.c_str())) {
                        function_adder_subgrids_to_wtiles = new runtime::Function(*modules[i], name_adder_subgrids_to_wtiles.c_str());
                    }
                    if (dlsym(*modules[i], kernel::cpu::name_splitter_subgrids_from_wtiles.c_str())) {
                        function_splitter_subgrids_from_wtiles = new runtime::Function(*modules[i], name_splitter_subgrids_from_wtiles.c_str());
                    }
                } // end for
            } // end load_kernel_funcions

            // Function signatures
            #define sig_gridder                       (void (*)(int,int,int,float,float,const float*,int,int,void*,void*,void*,void*,void*,void*,void*,void*,void*))
            #define sig_degridder                     (void (*)(int,int,int,float,float,const float*,int,int,void*,void*,void*,void*,void*,void*,void*,void*))

            #define sig_calibrate                     (void (*)(\
                                                          const unsigned int               nr_subgrids,\
                                                          const unsigned int               grid_size,\
                                                          const unsigned int               subgrid_size,\
                                                          const float                      image_size,\
                                                          const float                      w_step_in_lambda,\
                                                          const float* __restrict__        shift,\
                                                          const unsigned int               max_nr_timesteps,\
                                                          const unsigned int               nr_channels,\
                                                          const unsigned int               nr_stations,\
                                                          const unsigned int               nr_terms,\
                                                          const unsigned int               nr_time_slots,\
                                                          const idg::UVW<float>*           uvw,\
                                                          const float*                     wavenumbers,\
                                                                idg::float2*               visibilities,\
                                                          const float*                     weights,\
                                                          const idg::float2*               aterms,\
                                                          const idg::float2*               aterm_derivatives,\
                                                          const int*                       aterms_indices,\
                                                          const idg::Metadata*             metadata,\
                                                          const idg::float2*               subgrid,\
                                                          const idg::float2*               phasors,\
                                                          double*                          hessian,\
                                                          double*                          gradient, \
                                                          double*                          residual))

            #define sig_calibrate_hessian_vector_product1 (void (*)(\
                                                          const unsigned int               nr_subgrids,\
                                                          const unsigned int               grid_size,\
                                                          const unsigned int               subgrid_size,\
                                                          const float                      image_size,\
                                                          const float                      w_step_in_lambda,\
                                                          const float* __restrict__        shift,\
                                                          const unsigned int               max_nr_timesteps,\
                                                          const unsigned int               nr_channels,\
                                                          const unsigned int               nr_stations,\
                                                          const unsigned int               nr_terms,\
                                                          const unsigned int               nr_time_slots,\
                                                          const idg::UVW<float>*           uvw,\
                                                          const float*                     wavenumbers,\
                                                                idg::float2*               visibilities,\
                                                          const float*                     weights,\
                                                          const idg::float2*               aterms,\
                                                          const idg::float2*               aterm_derivatives,\
                                                          const int*                       aterms_indices,\
                                                          const idg::Metadata*             metadata,\
                                                          const idg::float2*               subgrid,\
                                                          const idg::float2*               phasors,\
                                                          const float*                     parameter_vector))

            #define sig_calibrate_hessian_vector_product2 (void (*)())

            #define sig_phasor                        (void (*)(int,int,int,float,float,const float*,int,int,void*,void*,void*,void*))
            #define sig_fft		                      (void (*)(long,long,long,void*,int))
            #define sig_adder	                      (void (*)(long,long,int,void*,void*,void*))
            #define sig_splitter                      (void (*)(long,long,int,void*,void*,void*))
            #define sig_adder_wstack                  (void (*)(long,long,int,void*,void*,void*))
            #define sig_splitter_wstack               (void (*)(long,long,int,void*,void*,void*))
            #define sig_adder_subgrids_to_wtiles      (void (*)(long,int,int,void*,void*,void*))
            #define sig_adder_wtiles_to_grid          (void (*)(int,int,float,float,int,void*,void*,void*,void*))
            #define sig_splitter_subgrids_from_wtiles (void (*)(long,int,int,void*,void*,void*))
            #define sig_splitter_wtiles_from_grid     (void (*)(int,int,float,float,int,void*,void*,void*,void*))

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
                void *aterm_idx,
                void *avg_aterm,
                void *metadata,
                void *subgrid)
            {
                powersensor::State states[2];
                states[0] = powerSensor->read();
                (sig_gridder (void *) *function_gridder)(
                  nr_subgrids, grid_size, subgrid_size, image_size, w_step, shift, nr_channels, nr_stations,
                  uvw, wavenumbers, visibilities, spheroidal, aterm, aterm_idx, avg_aterm, metadata, subgrid);
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
                void *aterm_idx,
                void *metadata,
                void *subgrid)
            {
                powersensor::State states[2];
                states[0] = powerSensor->read();
                (sig_degridder (void *) *function_degridder)(
                  nr_subgrids, grid_size, subgrid_size, image_size, w_step, shift, nr_channels, nr_stations,
                  uvw, wavenumbers, visibilities, spheroidal, aterm, aterm_idx, metadata, subgrid);
                states[1] = powerSensor->read();
                if (report) { report->update_degridder(states[0], states[1]); }
            }

            void InstanceCPU::run_calibrate(
                int nr_subgrids,
                int grid_size,
                int subgrid_size,
                float image_size,
                float w_step,
                const float* shift,
                int max_nr_timesteps,
                int nr_channels,
                int nr_terms,
                int nr_stations,
                int nr_time_slots,
                const idg::UVW<float> *uvw,
                const float *wavenumbers,
                idg::float2 *visibilities,
                const float *weights,
                const idg::float2 *aterm,
                const idg::float2 *aterm_derivative,
                const int* aterms_indices,
                const idg::Metadata *metadata,
                const idg::float2 *subgrid,
                const idg::float2 *phasors,
                double *hessian,
                double *gradient,
                double *residual)
            {
                powersensor::State states[2];
                states[0] = powerSensor->read();
                (sig_calibrate (void *) *function_calibrate)(
                  nr_subgrids, grid_size, subgrid_size, image_size, w_step, shift, max_nr_timesteps, nr_channels, nr_stations, nr_terms, nr_time_slots,
                  uvw, wavenumbers, visibilities, weights, aterm, aterm_derivative, aterms_indices, metadata, subgrid, phasors, hessian, gradient, residual);
                states[1] = powerSensor->read();
                if (report) { report->update_calibrate(states[0], states[1]); }
                if (report) { report->update_host(states[0], states[1]); }
            }

            void InstanceCPU::run_calibrate_hessian_vector_product1(
                const int station_nr,
                const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                const Array4D<Matrix2x2<std::complex<float>>>& derivative_aterms,
                const Array2D<float>& parameter_vector)
            {
                powersensor::State states[2];
                states[0] = powerSensor->read();
//                 (sig_calibrate_hessian_vector_product1 (void *) *function_calibrate_hessian_vector_product1)();
                states[1] = powerSensor->read();
                if (report) { report->update_calibrate(states[0], states[1]); }
                if (report) { report->update_host(states[0], states[1]); }
            }

            void InstanceCPU::run_calibrate_hessian_vector_product2(
                const int station_nr,
                const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                const Array4D<Matrix2x2<std::complex<float>>>& derivative_aterms,
                Array2D<float>& parameter_vector)
            {
                powersensor::State states[2];
                states[0] = powerSensor->read();
                (sig_calibrate_hessian_vector_product2 (void *) *function_calibrate_hessian_vector_product2)();
                states[1] = powerSensor->read();
                if (report) { report->update_calibrate(states[0], states[1]); }
                if (report) { report->update_host(states[0], states[1]); }
            }

            void InstanceCPU::run_phasor(
                int nr_subgrids,
                int grid_size,
                int subgrid_size,
                float image_size,
                float w_step,
                const float* shift,
                int max_nr_timesteps,
                int nr_channels,
                void *uvw,
                void *wavenumbers,
                void *metadata,
                void *phasors)
            {
                (sig_phasor (void *) *function_phasor)(
                  nr_subgrids, grid_size, subgrid_size, image_size, w_step, shift,
                  max_nr_timesteps, nr_channels, uvw, wavenumbers, metadata, phasors);
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

        void InstanceCPU::run_adder_subgrids_to_wtiles(
                int nr_subgrids,
                int grid_size,
                int subgrid_size,
                void *metadata,
                void *subgrid,
                void *tiles)
            {
                powersensor::State states[2];
                states[0] = powerSensor->read();
                (sig_adder_subgrids_to_wtiles (void *) *function_adder_subgrids_to_wtiles)(nr_subgrids, grid_size, subgrid_size, metadata, subgrid, tiles);
                states[1] = powerSensor->read();
                if (report) { report->update_adder(states[0], states[1]); }
            }

            void InstanceCPU::run_splitter_subgrids_from_wtiles(
                int nr_subgrids,
                int grid_size,
                int subgrid_size,
                void *metadata,
                void *subgrid,
                void *tiles)
            {
                powersensor::State states[2];
                states[0] = powerSensor->read();
                (sig_splitter_subgrids_from_wtiles (void *) *function_splitter_subgrids_from_wtiles)(nr_subgrids, grid_size, subgrid_size, metadata, subgrid, tiles);
                states[1] = powerSensor->read();
                if (report) { report->update_adder(states[0], states[1]); }
            }

            void InstanceCPU::run_adder_wtiles_to_grid(
                int grid_size,
                int subgrid_size,
                float image_size,
                float w_step,
                int nr_tiles,
                void *tile_ids,
                void *tile_coordinates,
                void *tiles,
                void *grid)
            {
                powersensor::State states[2];
                states[0] = powerSensor->read();
                (sig_adder_wtiles_to_grid (void *) *function_adder_wtiles_to_grid)(grid_size, subgrid_size, image_size, w_step, nr_tiles, tile_ids, tile_coordinates, tiles, grid);
                states[1] = powerSensor->read();
                if (report) { report->update_adder(states[0], states[1]); }

            }

            void InstanceCPU::run_splitter_wtiles_from_grid(
                int grid_size,
                int subgrid_size,
                float image_size,
                float w_step,
                int nr_tiles,
                void *tile_ids,
                void *tile_coordinates,
                void *tiles,
                void *grid)
            {
                powersensor::State states[2];
                states[0] = powerSensor->read();
                (sig_splitter_wtiles_from_grid (void *) *function_splitter_wtiles_from_grid)(grid_size, subgrid_size, image_size, w_step, nr_tiles, tile_ids, tile_coordinates, tiles, grid);
                states[1] = powerSensor->read();
                if (report) { report->update_adder(states[0], states[1]); }

            }

            void InstanceCPU::run_splitter_wtiles(
                unsigned int nr_subgrids,
                unsigned int grid_size,
                unsigned int subgrid_size,
                float image_size,
                float w_step,
                int subgrid_offset,
                WTileUpdateSet& wtile_initialize_set,
                void *wtiles,
                void *metadata,
                void *subgrid,
                void *grid)
            {
                for (unsigned int subgrid_index = 0; subgrid_index < nr_subgrids; ) {
                    unsigned int subgrid_index_front = wtile_initialize_set.front().subgrid_index;
                    if (subgrid_index_front == (subgrid_index + subgrid_offset))
                    {
                        wtile_initialize_set.pop_front();
                        subgrid_index_front = wtile_initialize_set.front().subgrid_index;
                        WTileUpdateInfo &wtile_initialize_info = wtile_initialize_set.front();
                        run_splitter_wtiles_from_grid(
                            grid_size,
                            subgrid_size,
                            image_size,
                            w_step,
                            wtile_initialize_info.wtile_ids.size(),
                            wtile_initialize_info.wtile_ids.data(),
                            wtile_initialize_info.wtile_coordinates.data(),
                            wtiles,
                            grid);
                    }

                    unsigned int nr_subgrids_ = nr_subgrids - subgrid_index;
                    if (subgrid_index_front - (subgrid_offset + subgrid_index) < nr_subgrids_)
                    {
                        nr_subgrids_ = subgrid_index_front - (subgrid_offset + subgrid_index);
                    }
                    if (nr_subgrids_ == 0) throw 1;
                    run_splitter_subgrids_from_wtiles(
                        nr_subgrids_,
                        grid_size,
                        subgrid_size,
                        &static_cast<Metadata*>(metadata)[subgrid_index],
                        &static_cast<std::complex<float>*>(subgrid)[subgrid_index * subgrid_size * subgrid_size * NR_CORRELATIONS],
                        wtiles);

                    subgrid_index += nr_subgrids_;
                } // end for subgrid_index
            } // end run_splitter_wtiles

        } // namespace cpu
    } // namespace kernel
} // namespace idg
