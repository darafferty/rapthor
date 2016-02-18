#include <cstdio> // remove()
#include <cstdlib>  // rand()
#include <ctime> // time() to init srand()
#include <complex>
#include <sstream>
#include <memory>
#include <dlfcn.h> // dlsym()
#include <omp.h> // omp_get_wtime
#include <libgen.h> // dirname() and basename()

#include "idg-config.h"
#include "OpenCL.h"

using namespace std;

namespace idg {
    namespace proxy {
        namespace hybrid {

            /// Constructors
            OpenCL::OpenCL(
                Parameters params) :
                cpu(params), opencl(params)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                cout << params;
                #endif

                mParams = params;

                // Initialize power sensor
                #if defined(MEASURE_POWER_ARDUINO)
                const char *str_power_sensor = getenv("POWER_SENSOR");
                if (!str_power_sensor) str_power_sensor = POWER_SENSOR;
                const char *str_power_file = getenv("POWER_FILE");
                if (!str_power_file) str_power_file = POWER_FILE;
                cout << "Opening power sensor: " << str_power_sensor << endl;
                cout << "Writing power consumption to file: " << str_power_file << endl;
                opencl::powerSensor.init(str_power_sensor, str_power_file);
                #else
                opencl::powerSensor.init();
                #endif
            }

            /// Destructor
            OpenCL::~OpenCL() {
            }

            /*
                High level routines
                These routines operate on grids
            */
            void OpenCL::grid_visibilities(
                const std::complex<float> *visibilities,
                const float *uvw,
                const float *wavenumbers,
                const int *baselines,
                std::complex<float> *grid,
                const float w_offset,
                const int kernel_size,
                const std::complex<float> *aterm,
                const int *aterm_offsets,
                const float *spheroidal) {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Constants
                auto nr_stations = mParams.get_nr_stations();
                auto nr_baselines = mParams.get_nr_baselines();
                auto nr_time = mParams.get_nr_time();
                auto nr_timeslots = mParams.get_nr_timeslots();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto gridsize = mParams.get_grid_size();
                auto subgridsize = mParams.get_subgrid_size();
                auto jobsize = mParams.get_job_size_gridder();

                // Load kernels
                unique_ptr<idg::kernel::opencl::Gridder> kernel_gridder = opencl.get_kernel_gridder();
                unique_ptr<idg::kernel::cpu::Adder> kernel_adder = cpu.get_kernel_adder();
                unique_ptr<idg::kernel::cpu::GridFFT> kernel_fft = cpu.get_kernel_fft();

                // Load context and device
                cl::Context context = opencl.get_context();
                cl::Device device = opencl.get_device();

                // Initialize metadata
                auto max_nr_timesteps = kernel_gridder->get_max_nr_timesteps();
                auto plan = create_plan(uvw, wavenumbers, baselines,
                                        aterm_offsets, kernel_size,
                                        max_nr_timesteps);
                auto nr_subgrids = plan.get_nr_subgrids();
                const Metadata *metadata = plan.get_metadata_ptr();

                // Initialize
                cl::CommandQueue executequeue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
                cl::CommandQueue htodqueue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
                cl::CommandQueue dtohqueue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
                const int nr_streams = 3;

                // Host memory
                cl::Buffer h_visibilities(context, CL_MEM_ALLOC_HOST_PTR, opencl.sizeof_visibilities(nr_baselines));
                cl::Buffer h_uvw(context, CL_MEM_ALLOC_HOST_PTR, opencl.sizeof_uvw(nr_baselines));
                cl::Buffer h_metadata(context, CL_MEM_ALLOC_HOST_PTR, opencl.sizeof_metadata(nr_subgrids));
                cl::Buffer h_subgrids(context, CL_MEM_ALLOC_HOST_PTR, opencl.sizeof_subgrids(nr_subgrids));
                void *subgrids = htodqueue.enqueueMapBuffer(h_subgrids, CL_FALSE, CL_MAP_READ, 0, opencl.sizeof_subgrids(nr_subgrids));

                // Copy input data to host memory
                htodqueue.enqueueWriteBuffer(h_visibilities, CL_FALSE, 0,  opencl.sizeof_visibilities(nr_baselines), visibilities);
                htodqueue.enqueueWriteBuffer(h_uvw, CL_FALSE, 0,  opencl.sizeof_uvw(nr_baselines), uvw);
                htodqueue.enqueueWriteBuffer(h_metadata, CL_FALSE, 0,  opencl.sizeof_metadata(nr_subgrids), metadata);

                // Device memory
                cl::Buffer d_wavenumbers = cl::Buffer(context, CL_MEM_READ_WRITE, opencl.sizeof_wavenumbers());
                cl::Buffer d_aterm       = cl::Buffer(context, CL_MEM_READ_WRITE, opencl.sizeof_aterm());
                cl::Buffer d_spheroidal  = cl::Buffer(context, CL_MEM_READ_WRITE, opencl.sizeof_spheroidal());

                // Performance measurements
                double total_runtime_gridder = 0;
                double total_runtime_fft = 0;
                double total_runtime_scaler = 0;
                double total_runtime_adder = 0;
                PowerSensor::State startState = opencl::powerSensor.read();

                // Copy static device memory
                htodqueue.enqueueWriteBuffer(d_wavenumbers, CL_FALSE, 0, opencl.sizeof_wavenumbers(), wavenumbers);
                htodqueue.enqueueWriteBuffer(d_aterm, CL_FALSE, 0, opencl.sizeof_aterm(), aterm);
                htodqueue.enqueueWriteBuffer(d_spheroidal, CL_FALSE, 0, opencl.sizeof_spheroidal(), spheroidal);

                // Start gridder
                #pragma omp parallel num_threads(nr_streams)
                {
                    // Events
                    vector<cl::Event> inputReady(1), computeReady(1), outputReady(1);

                    // Private device memory
                    auto max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);
                    cl::Buffer d_visibilities = cl::Buffer(context, CL_MEM_READ_WRITE, opencl.sizeof_visibilities(jobsize));
                    cl::Buffer d_uvw          = cl::Buffer(context, CL_MEM_READ_WRITE, opencl.sizeof_uvw(jobsize));
                    cl::Buffer d_subgrids = cl::Buffer(context, CL_MEM_READ_WRITE, opencl.sizeof_subgrids(max_nr_subgrids));
                    cl::Buffer d_metadata = cl::Buffer(context, CL_MEM_READ_WRITE, opencl.sizeof_metadata(max_nr_subgrids));

                    // Performance counters
                    PerformanceCounter counters[4];
                    #if defined(MEASURE_POWER_ARDUINO)
                    for (int i = 0; i < 4; i++) {
                        counters[i].setPowerSensor(&opencl::powerSensor);
                    }
                    #endif

                    // Power measurement
                    LikwidPowerSensor::State powerStates[3];

                    #pragma omp for schedule(dynamic)
                    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                        // Compute the number of baselines to process in current iteration
                        int current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;

                        // Number of subgrids for all baselines in job
                        auto current_nr_subgrids = plan.get_nr_subgrids(bl, current_nr_baselines);

                        // Offsets
                        size_t uvw_offset          = bl * opencl.sizeof_uvw(1);
                        size_t visibilities_offset = bl * opencl.sizeof_visibilities(1);
                        size_t metadata_offset     = bl * opencl.sizeof_metadata(1);
                        size_t subgrid_offset      = bl * opencl.sizeof_subgrids(1);

                        // Number of subgrids for all baselines in job
                        auto subgrid_elements      = subgridsize * subgridsize * nr_polarizations;

                        // Get pointers
                        void *metadata_ptr = (void *) plan.get_metadata_ptr(bl);
                        void *subgrids_ptr = subgrids + subgrid_elements*plan.get_subgrid_offset(bl);

                        #pragma omp critical (GPU)
                        {
                            // Copy input data to device
                            htodqueue.enqueueCopyBuffer(h_uvw, d_uvw, uvw_offset, 0, opencl.sizeof_uvw(current_nr_baselines), NULL, NULL);
                            htodqueue.enqueueCopyBuffer(h_visibilities, d_visibilities, visibilities_offset, 0, opencl.sizeof_visibilities(current_nr_baselines), NULL, NULL);
                            htodqueue.enqueueCopyBuffer(h_metadata, d_metadata, metadata_offset, 0, opencl.sizeof_metadata(current_nr_subgrids), NULL, NULL);
                            htodqueue.enqueueMarkerWithWaitList(NULL, &inputReady[0]);

                            // Launch gridder kernel
                            executequeue.enqueueMarkerWithWaitList(&inputReady, NULL);
                            kernel_gridder->launchAsync(
                                executequeue, current_nr_baselines, current_nr_subgrids, w_offset, d_uvw, d_wavenumbers,
                                d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids, counters[0]);
                            executequeue.enqueueMarkerWithWaitList(NULL, &computeReady[0]);

                            // Copy subgrid to host
                            dtohqueue.enqueueBarrierWithWaitList(&computeReady, NULL);
                            dtohqueue.enqueueCopyBuffer(d_subgrids, h_subgrids, 0, subgrid_offset, opencl.sizeof_subgrids(current_nr_subgrids), NULL, &outputReady[0]);
                        }

                        outputReady[0].wait();

                        // Run fft
                        powerStates[0] = cpu.read_power();
                        kernel_fft->run(subgridsize, current_nr_subgrids, subgrids_ptr, FFTW_BACKWARD);
                        powerStates[1] = cpu.read_power();

                        // Run adder
                        #pragma omp critical (CPU)
                        {
                            kernel_adder->run(current_nr_subgrids, metadata_ptr, subgrids_ptr, grid);
                        }
                        powerStates[2] = cpu.read_power();
                        auxiliary::report("    fft", LikwidPowerSensor::seconds(powerStates[0], powerStates[1]),
                                                     kernel_fft->flops(subgridsize, current_nr_subgrids),
                                                     kernel_fft->bytes(subgridsize, current_nr_subgrids),
                                                     LikwidPowerSensor::Watt(powerStates[0], powerStates[1]));
                        auxiliary::report("  adder", LikwidPowerSensor::seconds(powerStates[1], powerStates[2]),
                                                     kernel_adder->flops(current_nr_subgrids),
                                                     kernel_adder->bytes(current_nr_subgrids),
                                                     LikwidPowerSensor::Watt(powerStates[1], powerStates[2]));
                    }
                }

                // Unmap subgrids
                dtohqueue.enqueueUnmapMemObject(h_subgrids, subgrids);

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                PowerSensor::State stopState  = opencl::powerSensor.read();
                uint64_t total_flops_gridder  = kernel_gridder->flops(nr_baselines, nr_subgrids);
                uint64_t total_bytes_gridder  = kernel_gridder->bytes(nr_baselines, nr_subgrids);
                uint64_t total_flops_fft      = kernel_fft->flops(subgridsize, nr_subgrids);
                uint64_t total_bytes_fft      = kernel_fft->bytes(subgridsize, nr_subgrids);
                uint64_t total_flops_adder    = kernel_adder->flops(nr_subgrids);
                uint64_t total_bytes_adder    = kernel_adder->bytes(nr_subgrids);
                uint64_t total_flops_gridding = total_flops_gridder + total_flops_fft + total_flops_adder;
                uint64_t total_bytes_gridding = total_bytes_gridder + total_bytes_fft + total_bytes_adder;
                double total_runtime_gridding = PowerSensor::seconds(startState, stopState);
                double total_watt_gridding    = PowerSensor::Watt(startState, stopState);
                auxiliary::report("|gridding", total_runtime_gridding, total_flops_gridding, total_bytes_gridding, total_watt_gridding);
                auxiliary::report_visibilities("|gridding", total_runtime_gridding, nr_baselines, nr_time, nr_channels);
                clog << endl;
                #endif
            }

            void OpenCL::degrid_visibilities(
                std::complex<float> *visibilities,
                const float *uvw,
                const float *wavenumbers,
                const int *baselines,
                const std::complex<float> *grid,
                const float w_offset,
                const int kernel_size,
                const std::complex<float> *aterm,
                const int *aterm_offsets,
                const float *spheroidal) {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
                // Constants
                auto nr_stations = mParams.get_nr_stations();
                auto nr_baselines = mParams.get_nr_baselines();
                auto nr_time = mParams.get_nr_time();
                auto nr_timeslots = mParams.get_nr_timeslots();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto gridsize = mParams.get_grid_size();
                auto subgridsize = mParams.get_subgrid_size();
                auto jobsize = mParams.get_job_size_degridder();

                // Load kernels
                unique_ptr<idg::kernel::opencl::Degridder> kernel_degridder = opencl.get_kernel_degridder();
                unique_ptr<idg::kernel::cpu::Splitter> kernel_splitter = cpu.get_kernel_splitter();
                unique_ptr<idg::kernel::cpu::GridFFT> kernel_fft = cpu.get_kernel_fft();

                // Load context and device
                cl::Context context = opencl.get_context();
                cl::Device device = opencl.get_device();

                // Initialize metadata
                auto max_nr_timesteps = kernel_degridder->get_max_nr_timesteps();
                auto plan = create_plan(uvw, wavenumbers, baselines,
                                        aterm_offsets, kernel_size,
                                        max_nr_timesteps);
                auto nr_subgrids = plan.get_nr_subgrids();
                const Metadata *metadata = plan.get_metadata_ptr();

                // Initialize
                cl::CommandQueue executequeue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
                cl::CommandQueue htodqueue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
                cl::CommandQueue dtohqueue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
                const int nr_streams = 3;

                // Host memory
                cl::Buffer h_visibilities(context, CL_MEM_ALLOC_HOST_PTR, opencl.sizeof_visibilities(nr_baselines));
                cl::Buffer h_uvw(context, CL_MEM_ALLOC_HOST_PTR, opencl.sizeof_uvw(nr_baselines));
                cl::Buffer h_metadata(context, CL_MEM_ALLOC_HOST_PTR, opencl.sizeof_metadata(nr_subgrids));
                cl::Buffer h_subgrids(context, CL_MEM_ALLOC_HOST_PTR, opencl.sizeof_subgrids(nr_subgrids));
                void *subgrids = htodqueue.enqueueMapBuffer(h_subgrids, CL_FALSE, CL_MAP_WRITE, 0, opencl.sizeof_subgrids(nr_subgrids));

                // Copy input data to host memory
                htodqueue.enqueueWriteBuffer(h_uvw, CL_FALSE, 0,  opencl.sizeof_uvw(nr_baselines), uvw);
                htodqueue.enqueueWriteBuffer(h_metadata, CL_FALSE, 0,  opencl.sizeof_metadata(nr_subgrids), metadata);

                // Device memory
                cl::Buffer d_wavenumbers = cl::Buffer(context, CL_MEM_READ_WRITE, opencl.sizeof_wavenumbers());
                cl::Buffer d_aterm       = cl::Buffer(context, CL_MEM_READ_WRITE, opencl.sizeof_aterm());
                cl::Buffer d_spheroidal  = cl::Buffer(context, CL_MEM_READ_WRITE, opencl.sizeof_spheroidal());

                // Performance measurements
                double total_runtime_degridder = 0;
                double total_runtime_fft = 0;
                double total_runtime_scaler = 0;
                double total_runtime_splitter = 0;
                PowerSensor::State startState = opencl::powerSensor.read();

                // Copy static device memory
                htodqueue.enqueueWriteBuffer(d_wavenumbers, CL_FALSE, 0, opencl.sizeof_wavenumbers(), wavenumbers);
                htodqueue.enqueueWriteBuffer(d_aterm, CL_FALSE, 0, opencl.sizeof_aterm(), aterm);
                htodqueue.enqueueWriteBuffer(d_spheroidal, CL_FALSE, 0, opencl.sizeof_spheroidal(), spheroidal);

                // Start degridder
                #pragma omp parallel num_threads(nr_streams)
                {
                    // Events
                    vector<cl::Event> inputReady(1), computeReady(1), outputReady(1);

                    // Private device memory
                    auto max_nr_subgrids = plan.get_max_nr_subgrids(0, nr_baselines, jobsize);
                    cl::Buffer d_visibilities = cl::Buffer(context, CL_MEM_READ_WRITE, opencl.sizeof_visibilities(jobsize));
                    cl::Buffer d_uvw          = cl::Buffer(context, CL_MEM_READ_WRITE, opencl.sizeof_uvw(jobsize));
                    cl::Buffer d_subgrids = cl::Buffer(context, CL_MEM_READ_WRITE, opencl.sizeof_subgrids(max_nr_subgrids));
                    cl::Buffer d_metadata = cl::Buffer(context, CL_MEM_READ_WRITE, opencl.sizeof_metadata(max_nr_subgrids));

                    // Performance counters
                    PerformanceCounter counters[4];
                    #if defined(MEASURE_POWER_ARDUINO)
                    for (int i = 0; i < 4; i++) {
                        counters[i].setPowerSensor(&opencl::powerSensor);
                    }
                    #endif

                    // Power measurement
                    LikwidPowerSensor::State powerStates[3];

                    #pragma omp for schedule(dynamic)
                    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                        // Compute the number of baselines to process in current iteration
                        int current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;

                        // Number of subgrids for all baselines in job
                        auto current_nr_subgrids = plan.get_nr_subgrids(bl, current_nr_baselines);

                        // Offsets
                        size_t uvw_offset          = bl * opencl.sizeof_uvw(1);
                        size_t visibilities_offset = bl * opencl.sizeof_visibilities(1);
                        size_t metadata_offset     = bl * opencl.sizeof_metadata(1);
                        size_t subgrid_offset      = bl * opencl.sizeof_subgrids(1);

                        // Number of subgrids for all baselines in job
                        auto subgrid_elements      = subgridsize * subgridsize * nr_polarizations;

                        // Get pointers
                        void *metadata_ptr = (void *) plan.get_metadata_ptr(bl);
                        void *subgrids_ptr = subgrids + subgrid_elements*plan.get_subgrid_offset(bl);

                        // Run splitter
                        powerStates[0] = cpu.read_power();
                        kernel_splitter->run(current_nr_subgrids, metadata_ptr, subgrids_ptr, (void *) grid);
                        powerStates[1] = cpu.read_power();

                         // Run fft
                        kernel_fft->run(subgridsize, current_nr_subgrids, subgrids_ptr, FFTW_BACKWARD);
                        powerStates[2] = cpu.read_power();

                        #pragma omp critical (GPU)
                        {
                            // Copy input data to device
                            htodqueue.enqueueCopyBuffer(h_uvw, d_uvw, uvw_offset, 0, opencl.sizeof_uvw(current_nr_baselines), NULL, NULL);
                            htodqueue.enqueueCopyBuffer(h_subgrids, d_subgrids, subgrid_offset, 0, opencl.sizeof_subgrids(current_nr_subgrids), NULL, NULL);
                            htodqueue.enqueueCopyBuffer(h_metadata, d_metadata, metadata_offset, 0, opencl.sizeof_metadata(current_nr_subgrids), NULL, NULL);
                            htodqueue.enqueueMarkerWithWaitList(NULL, &inputReady[0]);

                            // Launch degridder kernel
                            executequeue.enqueueMarkerWithWaitList(&inputReady, NULL);
                            kernel_degridder->launchAsync(
                                executequeue, current_nr_baselines, current_nr_subgrids, w_offset, d_uvw, d_wavenumbers,
                                d_visibilities, d_spheroidal, d_aterm, d_metadata, d_subgrids, counters[0]);
                            executequeue.enqueueMarkerWithWaitList(NULL, &computeReady[0]);

                            // Copy visibilities to host
                            dtohqueue.enqueueBarrierWithWaitList(&computeReady, NULL);
                            dtohqueue.enqueueCopyBuffer(d_visibilities, h_visibilities, 0, visibilities_offset, opencl.sizeof_visibilities(current_nr_baselines), NULL, &outputReady[0]);
                        }

                        outputReady[0].wait();

                        auxiliary::report(" splitter", LikwidPowerSensor::seconds(powerStates[0], powerStates[1]),
                                                       kernel_splitter->flops(current_nr_subgrids),
                                                       kernel_splitter->bytes(current_nr_subgrids),
                                                       LikwidPowerSensor::Watt(powerStates[0], powerStates[1]));
                        auxiliary::report("      fft", LikwidPowerSensor::seconds(powerStates[1], powerStates[2]),
                                                       kernel_fft->flops(subgridsize, current_nr_subgrids),
                                                       kernel_fft->bytes(subgridsize, current_nr_subgrids),
                                                       LikwidPowerSensor::Watt(powerStates[1], powerStates[2]));
                    }
                }

                // Copy visibilities from host memory
                htodqueue.enqueueReadBuffer(h_visibilities, CL_TRUE, 0,  opencl.sizeof_visibilities(nr_baselines), visibilities);

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                PowerSensor::State stopState  = opencl::powerSensor.read();
                uint64_t total_flops_degridder  = kernel_degridder->flops(nr_baselines, nr_subgrids);
                uint64_t total_bytes_degridder  = kernel_degridder->bytes(nr_baselines, nr_subgrids);
                uint64_t total_flops_fft      = kernel_fft->flops(subgridsize, nr_subgrids);
                uint64_t total_bytes_fft      = kernel_fft->bytes(subgridsize, nr_subgrids);
                uint64_t total_flops_splitter    = kernel_splitter->flops(nr_subgrids);
                uint64_t total_bytes_splitter    = kernel_splitter->bytes(nr_subgrids);
                uint64_t total_flops_degridding = total_flops_degridder + total_flops_fft + total_flops_splitter;
                uint64_t total_bytes_degridding = total_bytes_degridder + total_bytes_fft + total_bytes_splitter;
                double total_runtime_degridding = PowerSensor::seconds(startState, stopState);
                double total_watt_degridding    = PowerSensor::Watt(startState, stopState);
                auxiliary::report("|degridding", total_runtime_degridding, total_flops_degridding, total_bytes_degridding, total_watt_degridding);
                auxiliary::report_visibilities("|degridding", total_runtime_degridding, nr_baselines, nr_time, nr_channels);
                clog << endl;
                #endif
            }

            void OpenCL::transform(DomainAtoDomainB direction,
                std::complex<float>* grid) {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                cpu.transform(direction, grid);
            }

        } // namespace hybrid
    } // namespace proxy
} // namespace idg

// C interface:
// Rationale: calling the code from C code and Fortran easier,
// and bases to create interface to scripting languages such as
// Python, Julia, Matlab, ...
extern "C" {
    typedef idg::proxy::hybrid::OpenCL Hybrid_OpenCL;

    Hybrid_OpenCL* Hybrid_OpenCL_init(
                unsigned int nr_stations,
                unsigned int nr_channels,
                unsigned int nr_time,
                unsigned int nr_timeslots,
                float        imagesize,
                unsigned int grid_size,
                unsigned int subgrid_size)
    {
        idg::Parameters P;
        P.set_nr_stations(nr_stations);
        P.set_nr_channels(nr_channels);
        P.set_nr_time(nr_time);
        P.set_nr_timeslots(nr_timeslots);
        P.set_imagesize(imagesize);
        P.set_subgrid_size(subgrid_size);
        P.set_grid_size(grid_size);

        return new Hybrid_OpenCL(P);
    }

    void Hybrid_OpenCL_grid(Hybrid_OpenCL* p,
                            void *visibilities,
                            void *uvw,
                            void *wavenumbers,
                            void *metadata,
                            void *grid,
                            float w_offset,
                            int   kernel_size,
                            void *aterm,
                            void *aterm_offsets,
                            void *spheroidal)
    {
         p->grid_visibilities(
                (const std::complex<float>*) visibilities,
                (const float*) uvw,
                (const float*) wavenumbers,
                (const int*) metadata,
                (std::complex<float>*) grid,
                w_offset,
                kernel_size,
                (const std::complex<float>*) aterm,
                (const int*) aterm_offsets,
                (const float*) spheroidal);
    }

    void Hybrid_OpenCL_degrid(Hybrid_OpenCL* p,
                            void *visibilities,
                            void *uvw,
                            void *wavenumbers,
                            void *metadata,
                            void *grid,
                            float w_offset,
                            int   kernel_size,
                            void *aterm,
                            void *aterm_offsets,
                            void *spheroidal)
    {
         p->degrid_visibilities(
                (std::complex<float>*) visibilities,
                    (const float*) uvw,
                    (const float*) wavenumbers,
                    (const int*) metadata,
                    (const std::complex<float>*) grid,
                    w_offset,
                    kernel_size,
                    (const std::complex<float>*) aterm,
                    (const int*) aterm_offsets,
                    (const float*) spheroidal);
     }

    void Hybrid_OpenCL_transform(Hybrid_OpenCL* p,
                    int direction,
                    void *grid)
    {
       if (direction!=0)
           p->transform(idg::ImageDomainToFourierDomain,
                    (std::complex<float>*) grid);
       else
           p->transform(idg::FourierDomainToImageDomain,
                    (std::complex<float>*) grid);
    }

    void Hybrid_OpenCL_destroy(Hybrid_OpenCL* p) {
       delete p;
    }

}  // end extern "C"
