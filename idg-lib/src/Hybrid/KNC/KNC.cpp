#include <cstdio> // remove()
#include <complex>
#include <sstream>
#include <memory>
#include <omp.h> // omp_get_wtime

#include "idg-config.h"
#include "KNC.h"

using namespace std;

namespace idg {
    namespace proxy {
        namespace hybrid {

            // Power sensor (TODO: why here? why not a member of the class?)
            static PowerSensor *powerSensor = nullptr;

            /// Constructors
            KNC::KNC(Parameters params)
            {
                #if defined(DEBUG)
                cout << "KNC::" << __func__ << endl;
                cout << params;
                #endif

                mParams = params;

                #if defined(MEASURE_POWER_ARDUINO)
                cout << "Opening power sensor: " << STR_POWER_SENSOR << endl;
                cout << "Writing power consumption to file: " << STR_POWER_FILE << endl;
                powerSensor = new PowerSensor(STR_POWER_SENSOR, STR_POWER_FILE);
                const char *str_power_sensor = getenv("POWER_SENSOR");
                if (!str_power_sensor) str_power_sensor = STR_POWER_SENSOR;
                const char *str_power_file = getenv("POWER_FILE");
                if (!str_power_file) str_power_file = STR_POWER_FILE;
                #else
                powerSensor = new PowerSensor();
                // TODO: where is it freed? Need to be done in destructor!
                #endif
            }


            /// High level routines
            void KNC::grid_visibilities(
                const complex<float> *visibilities,
                const float *uvw,
                const float *wavenumbers,
                const int *baselines,
                complex<float> *grid,
                const float w_offset,
                const int kernel_size,
                const complex<float> *aterm,
                const int *aterm_offsets,
                const float *spheroidal)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                try {
                    // Proxy constants
                    auto nr_baselines = mParams.get_nr_baselines();
                    auto subgridsize = mParams.get_subgrid_size();
                    auto nr_polarizations = mParams.get_nr_polarizations();;
                    auto nr_time = mParams.get_nr_time();
                    auto nr_channels = mParams.get_nr_channels();

                    // Checks arguments
                    if (kernel_size <= 0 || kernel_size >= subgridsize-1) {
                        throw invalid_argument("0 < kernel_size < subgridsize-1 not true");
                    }

                    double runtime = -omp_get_wtime();

                    // Initialize metadata
                    auto plan = create_plan(uvw, wavenumbers, baselines,
                                            aterm_offsets, kernel_size);
                    auto nr_subgrids = plan.get_nr_subgrids();

                    // Allocate 'subgrids' memory for subgrids
                    auto size_subgrids = 1ULL*nr_subgrids*nr_polarizations*
                                        subgridsize*subgridsize;
                    auto subgrids = new complex<float>[size_subgrids];

                    runtime += omp_get_wtime();
                    #if defined (REPORT_TOTAL)
                    auxiliary::report("init", runtime);
                    #endif

                    runtime -= omp_get_wtime();

                    // Run subroutines
                    grid_onto_subgrids(
                        plan,
                        w_offset,
                        uvw,
                        wavenumbers,
                        visibilities,
                        spheroidal,
                        aterm,
                        subgrids);

                    add_subgrids_to_grid(
                        plan,
                        subgrids,
                        grid);

                    runtime += omp_get_wtime();

                    #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                    uint64_t flops_gridder  = kernel::flops_gridder(mParams, nr_baselines, nr_subgrids);
                    uint64_t bytes_gridder  = kernel::bytes_gridder(mParams, nr_baselines, nr_subgrids);
                    uint64_t flops_fft      = kernel::flops_fft(mParams, subgridsize, nr_subgrids);
                    uint64_t bytes_fft      = kernel::bytes_fft(mParams, subgridsize, nr_subgrids);
                    uint64_t flops_adder    = kernel::flops_adder(mParams, nr_subgrids);
                    uint64_t bytes_adder    = kernel::bytes_adder(mParams, nr_subgrids);
                    uint64_t flops_gridding = flops_gridder + flops_fft + flops_adder;
                    uint64_t bytes_gridding = bytes_gridder + bytes_fft + bytes_adder;
                    auxiliary::report("|gridding", runtime,
                        flops_gridding, bytes_gridding);
                    auxiliary::report_visibilities("|gridding",
                        runtime, nr_baselines, nr_time, nr_channels);
                    clog << endl;
                    #endif

                    delete[] subgrids;

                } catch (const invalid_argument& e) {
                    cerr << __func__ << ": invalid argument: "
                         << e.what() << endl;
                    exit(1);
                } catch (const exception& e) {
                    cerr << __func__ << ": caught exception: "
                         << e.what() << endl;
                    exit(2);
                } catch (...) {
                    cerr << __func__ << ": caught unknown exception" << endl;
                    exit(3);
                }
            }


            void KNC::degrid_visibilities(
                std::complex<float> *visibilities,
                const float *uvw,
                const float *wavenumbers,
                const int *baselines,
                const std::complex<float> *grid,
                const float w_offset,
                const int kernel_size,
                const std::complex<float> *aterm,
                const int *aterm_offsets,
                const float *spheroidal)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                try {
                    // Proxy constants
                    auto nr_baselines = mParams.get_nr_baselines();
                    auto subgridsize = mParams.get_subgrid_size();
                    auto nr_polarizations = mParams.get_nr_polarizations();;
                    auto nr_time = mParams.get_nr_time();
                    auto nr_channels = mParams.get_nr_channels();

                    // Checks arguments
                    if (kernel_size <= 0 || kernel_size >= subgridsize-1) {
                        throw invalid_argument("0 < kernel_size < subgridsize-1 not true");
                    }

                    double runtime = -omp_get_wtime();

                    // Initialize metadata
                    auto plan = create_plan(uvw, wavenumbers, baselines,
                                            aterm_offsets, kernel_size);
                    auto nr_subgrids = plan.get_nr_subgrids();

                    // Allocate 'subgrids' memory for subgrids
                    auto size_subgrids = 1ULL*nr_subgrids*nr_polarizations*
                                        subgridsize*subgridsize;
                    auto subgrids = new complex<float>[size_subgrids];

                    runtime += omp_get_wtime();
                    #if defined (REPORT_TOTAL)
                    auxiliary::report("init", runtime);
                    #endif

                    runtime -= omp_get_wtime();

                    // Run subroutines
                    split_grid_into_subgrids(
                         plan,
                         subgrids,
                         grid);

                    degrid_from_subgrids(
                        plan,
                        w_offset,
                        uvw,
                        wavenumbers,
                        visibilities,
                        spheroidal,
                        aterm,
                        subgrids);

                    runtime += omp_get_wtime();

                    #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                    uint64_t flops_degridder  = kernel::flops_degridder(mParams, nr_baselines, nr_subgrids);
                    uint64_t bytes_degridder  = kernel::bytes_degridder(mParams, nr_baselines, nr_subgrids);
                    uint64_t flops_fft        = kernel::flops_fft(mParams, subgridsize, nr_subgrids);
                    uint64_t bytes_fft        = kernel::bytes_fft(mParams, subgridsize, nr_subgrids);
                    uint64_t flops_splitter   = kernel::flops_splitter(mParams, nr_subgrids);
                    uint64_t bytes_splitter   = kernel::bytes_splitter(mParams, nr_subgrids);
                    uint64_t flops_degridding   = flops_degridder + flops_fft + flops_splitter;
                    uint64_t bytes_degridding   = bytes_degridder + bytes_fft + bytes_splitter;
                    auxiliary::report("|degridding", runtime,
                        flops_degridding, bytes_degridding);
                    auxiliary::report_visibilities("|degridding",
                        runtime, nr_baselines, nr_time, nr_channels);
                    clog << endl;
                    #endif

                    delete[] subgrids;

                } catch (const invalid_argument& e) {
                    cerr << __func__ << ": invalid argument: "
                         << e.what() << endl;
                    exit(1);
                } catch (const exception& e) {
                    cerr << __func__ << ": caught exception: "
                         << e.what() << endl;
                    exit(2);
                } catch (...) {
                    cerr << __func__ << ": caught unknown exception" << endl;
                    exit(3);
                }

            }



            void KNC::transform(DomainAtoDomainB direction, complex<float>* grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                cout << "Transform direction: " << direction << endl;
                #endif

                try {

                    int sign = (direction == FourierDomainToImageDomain) ? 1 : -1;

                    // Constants
                    auto gridsize = mParams.get_grid_size();
                    auto nr_polarizations = mParams.get_nr_polarizations();

                    double runtime = -omp_get_wtime();

                    if (direction == FourierDomainToImageDomain)
                        kernel::knc::ifftshift(nr_polarizations, gridsize, grid); // TODO: integrate into adder?
                    else
                        kernel::knc::ifftshift(nr_polarizations, gridsize, grid); // TODO: remove

                    // Start fft
                    #if defined(DEBUG)
                    cout << "FFT (direction: " << direction << ")" << endl;
                    #endif
                    kernel::knc::fft(gridsize, 1, grid, sign, nr_polarizations);

                    if (direction == FourierDomainToImageDomain)
                        kernel::knc::fftshift(nr_polarizations, gridsize, grid); // TODO: remove
                    else
                        kernel::knc::fftshift(nr_polarizations, gridsize, grid); // TODO: integrate into splitter?

                    runtime += omp_get_wtime();

                    #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                    auxiliary::report(">grid_fft", runtime,
                        kernel::flops_fft(mParams, gridsize, 1),
                        kernel::bytes_fft(mParams, gridsize, 1));
                    clog << endl;
                    #endif

                } catch (const exception& e) {
                    cerr << __func__ << " caught exception: "
                         << e.what() << endl;
                } catch (...) {
                    cerr << __func__ << " caught unknown exception" << endl;
                }
            }



            /// Low level routines
            void KNC::grid_onto_subgrids(
                const Plan& plan,
                const float w_offset,
                const float *uvw,
                const float *wavenumbers,
                const complex<float> *visibilities,
                const float *spheroidal,
                const complex<float> *aterm,
                complex<float> *subgrids)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Constants
                auto jobsize = mParams.get_job_size_gridder();
                auto nr_stations = mParams.get_nr_stations();
                auto nr_baselines = mParams.get_nr_baselines();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_time = mParams.get_nr_time();
                auto nr_timeslots = mParams.get_nr_timeslots();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
                auto gridsize = mParams.get_grid_size();
                auto imagesize = mParams.get_imagesize();

                // Number of elements in static
                int wavenumbers_elements = nr_channels;
                int spheroidal_elements  = subgridsize * subgridsize;
                int aterm_elements      = nr_stations * nr_timeslots *
                                           nr_polarizations * subgridsize *
                                           subgridsize;

                // Pointers to static data
                float *wavenumbers_ptr     = (float *) wavenumbers;
                float *spheroidal_ptr      = (float *) spheroidal;
                complex<float> *aterm_ptr = (complex<float> *) aterm;

                // Performance measurements
                double total_runtime_gridder = 0;
                double total_runtime_fft = 0;

                // Start gridder
                #pragma omp target data \
                        map(to:wavenumbers_ptr[0:wavenumbers_elements]) \
                        map(to:spheroidal_ptr[0:spheroidal_elements]) \
                        map(to:aterm_ptr[0:aterm_elements])
                {
                    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                        // Compute the number of baselines in current job
                        int current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;

                        // Number of elements in batch
                        int uvw_elements          = nr_time * sizeof(UVW)/sizeof(float);
                        int visibilities_elements = nr_time * nr_channels * nr_polarizations;
                        int metadata_elements     = sizeof(Metadata) / sizeof(int);
                        int subgrid_elements      = subgridsize * subgridsize * nr_polarizations;

                        // Number of subgrids for all baselines in job
                        auto current_nr_subgrids = plan.get_nr_subgrids(bl, current_nr_baselines);

                        // Pointers to data for current batch
                        float *uvw_ptr                   = (float *) uvw + bl * uvw_elements;
                        complex<float> *visibilities_ptr = (complex<float>*) visibilities + bl * visibilities_elements;
                        complex<float> *subgrids_ptr     = (complex<float>*) subgrids + subgrid_elements * plan.get_subgrid_offset(bl);
                        int *metadata_ptr                = (int *) plan.get_metadata_ptr(bl);

                        // Power measurement
                        PowerSensor::State powerStates[3];
                        #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                        powerStates[0] = powerSensor->read();
                        #endif

                        // Performance measurement
                        double runtime_gridder, runtime_fft;

                       #pragma omp target \
                               map(to:uvw_ptr[0:(current_nr_baselines * uvw_elements)]) \
                               map(to:visibilities_ptr[0:(current_nr_baselines * visibilities_elements)]) \
                               map(from:subgrids_ptr[0:(current_nr_subgrids * subgrid_elements)]) \
                               map(to:metadata_ptr[0:(current_nr_subgrids * metadata_elements)])
                        {
                            runtime_gridder = -omp_get_wtime();

                            kernel::knc::gridder(
                                current_nr_subgrids,
                                w_offset,
                                uvw_ptr,
                                wavenumbers_ptr,
                                visibilities_ptr,
                                spheroidal_ptr,
                                aterm_ptr,
                                metadata_ptr,
                                subgrids_ptr,
                                nr_stations,
                                nr_time,
                                nr_timeslots,
                                nr_channels,
                                gridsize,
                                subgridsize,
                                imagesize,
                                nr_polarizations);

                            runtime_gridder += omp_get_wtime();
                        }

                        #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                        powerStates[1] = powerSensor->read();
                        #endif

                        #pragma omp target \
                                map(tofrom:subgrids_ptr[0:(current_nr_subgrids * subgrid_elements)])
                        {
                            runtime_fft = -omp_get_wtime();

                            kernel::knc::fft(
                                subgridsize,
                                current_nr_subgrids,
                                subgrids_ptr,
                                1,
                                nr_polarizations);

                            runtime_fft += omp_get_wtime();
                        }

                        #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                        powerStates[2] = powerSensor->read();
                        total_runtime_gridder += runtime_gridder;
                        total_runtime_fft += runtime_fft;
                        #endif

                        #if defined(REPORT_VERBOSE)
                        auxiliary::report("gridder", runtime_gridder,
                            kernel::flops_gridder(mParams, current_nr_baselines, current_nr_subgrids),
                            kernel::bytes_gridder(mParams, current_nr_baselines, current_nr_subgrids),
                            PowerSensor::Watt(powerStates[0], powerStates[1]));
                        auxiliary::report("fft", runtime_fft,
                            kernel::flops_fft(mParams, subgridsize, current_nr_subgrids),
                            kernel::bytes_fft(mParams, subgridsize, current_nr_subgrids),
                            PowerSensor::Watt(powerStates[1], powerStates[2]));
                        #endif
                    } // end for bl

                } // end omp target data

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                clog << endl;
                auto nr_subgrids = plan.get_nr_subgrids();
                uint64_t total_flops_gridder  = kernel::flops_gridder(mParams, nr_baselines, nr_subgrids);
                uint64_t total_bytes_gridder  = kernel::bytes_gridder(mParams, nr_baselines, nr_subgrids);
                uint64_t total_flops_fft      = kernel::flops_fft(mParams, subgridsize, nr_subgrids);
                uint64_t total_bytes_fft      = kernel::bytes_fft(mParams, subgridsize, nr_subgrids);
                auxiliary::report("|gridder", total_runtime_gridder, total_flops_gridder,
                                  total_bytes_gridder);
                auxiliary::report("|fft", total_runtime_fft, total_flops_fft, total_bytes_fft);
                clog << endl;
                #endif
        }


            void KNC::add_subgrids_to_grid(
                const Plan& plan,
                const std::complex<float> *subgrids,
                std::complex<float> *grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Constants
                auto jobsize = mParams.get_job_size_adder();
                auto nr_baselines = mParams.get_nr_baselines();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
                auto gridsize = mParams.get_grid_size();

                // Performance measurements
                double total_runtime_adding = 0;
                double total_runtime_adder = 0;
                total_runtime_adding = -omp_get_wtime();

                // Run adder
                for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                    // Number of baselines in job
                    int current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;

                    // Number of elements in batch
                    auto current_nr_subgrids = plan.get_nr_subgrids(bl, current_nr_baselines);
                    auto elems_per_subgrid     = subgridsize * subgridsize
                                                 * nr_polarizations;

                    // Pointers to the first element in processed batch
                    void *subgrid_ptr  = const_cast<complex<float>*>(subgrids
                                         + elems_per_subgrid*plan.get_subgrid_offset(bl));
                    void *grid_ptr     = grid;
                    void *metadata_ptr = (void *) plan.get_metadata_ptr(bl);

                    double runtime_adder = -omp_get_wtime();
                    kernel::knc::adder(
                        current_nr_subgrids,
                        metadata_ptr,
                        subgrid_ptr,
                        grid_ptr,
                        gridsize,
                        subgridsize,
                        nr_polarizations);
                    runtime_adder += omp_get_wtime();

                    #if defined(REPORT_VERBOSE)
                    auxiliary::report("adder", runtime_adder,
                                      kernel::flops_adder(mParams, current_nr_subgrids),
                                      kernel::bytes_adder(mParams, current_nr_subgrids));
                    #endif
                    #if defined(REPORT_TOTAL)
                    total_runtime_adder += runtime_adder;
                    #endif
                } // end for bl

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                total_runtime_adding += omp_get_wtime();
                clog << endl;
                auto nr_subgrids = plan.get_nr_subgrids();
                uint64_t total_flops_adder = kernel::flops_adder(mParams, nr_subgrids);
                uint64_t total_bytes_adder = kernel::bytes_adder(mParams, nr_subgrids);
                auxiliary::report("|adder", total_runtime_adder, total_flops_adder, total_bytes_adder);
                auxiliary::report("|adding", total_runtime_adding, total_flops_adder, total_bytes_adder);
                auxiliary::report_subgrids("|adding", total_runtime_adding, nr_subgrids);
                clog << endl;
                #endif
            }


            void KNC::split_grid_into_subgrids(
                const Plan& plan,
                std::complex<float> *subgrids,
                const std::complex<float> *grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Constants
                auto jobsize = mParams.get_job_size_splitter();
                auto nr_baselines = mParams.get_nr_baselines();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
                auto gridsize = mParams.get_grid_size();

                // Performance measurements
                double total_runtime_splitting = 0;
                double total_runtime_splitter = 0;
                total_runtime_splitting = -omp_get_wtime();

                // Run splitter
                for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                    // Number of baselines in job
                    int current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;

                    // Number of elements in batch
                    auto current_nr_subgrids = plan.get_nr_subgrids(bl, current_nr_baselines);
                    auto elems_per_subgrid     = subgridsize * subgridsize * nr_polarizations;

                    // Pointers to the first element in processed batch
                    void *subgrid_ptr  = subgrids + elems_per_subgrid*plan.get_subgrid_offset(bl);
                    void *grid_ptr     = const_cast<complex<float>*>(grid);
                    void *metadata_ptr = (void *) plan.get_metadata_ptr(bl);

                    double runtime_splitter = -omp_get_wtime();
                    kernel::knc::splitter(
                        current_nr_subgrids,
                        metadata_ptr,
                        subgrid_ptr,
                        grid_ptr,
                        gridsize,
                        subgridsize,
                        nr_polarizations);
                    runtime_splitter += omp_get_wtime();

                    #if defined(REPORT_VERBOSE)
                    auxiliary::report("splitter", runtime_splitter,
                                      kernel::flops_splitter(mParams, current_nr_subgrids),
                                      kernel::bytes_splitter(mParams, current_nr_subgrids));
                    #endif
                    #if defined(REPORT_TOTAL)
                    total_runtime_splitter += runtime_splitter;
                    #endif
                } // end for bl

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                total_runtime_splitting += omp_get_wtime();
                clog << endl;
                auto nr_subgrids = plan.get_nr_subgrids();
                uint64_t total_flops_splitter = kernel::flops_splitter(mParams, nr_subgrids);
                uint64_t total_bytes_splitter = kernel::bytes_splitter(mParams, nr_subgrids);
                auxiliary::report("|splitter", total_runtime_splitter,
                                  total_flops_splitter, total_bytes_splitter);
                auxiliary::report("|splitting", total_runtime_splitting,
                                  total_flops_splitter, total_bytes_splitter);
                auxiliary::report_subgrids("|splitting", total_runtime_splitting,
                                           nr_subgrids);
                clog << endl;
                #endif

            }


            void KNC::degrid_from_subgrids(
                const Plan& plan,
                const float w_offset,
                const float *uvw,
                const float *wavenumbers,
                std::complex<float> *visibilities,
                const float *spheroidal,
                const std::complex<float> *aterm,
                const std::complex<float> *subgrids)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Constants
                auto jobsize = mParams.get_job_size_degridder();
                auto nr_stations = mParams.get_nr_stations();
                auto nr_baselines = mParams.get_nr_baselines();
                auto nr_channels = mParams.get_nr_channels();
                auto nr_time = mParams.get_nr_time();
                auto nr_timeslots = mParams.get_nr_timeslots();
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto subgridsize = mParams.get_subgrid_size();
                auto gridsize = mParams.get_grid_size();
                auto imagesize = mParams.get_imagesize();

                // Number of elements in static
                int wavenumbers_elements = nr_channels;
                int spheroidal_elements  = subgridsize * subgridsize;
                int aterm_elements      = nr_stations * nr_timeslots *
                                           nr_polarizations * subgridsize * subgridsize;

                // Pointers to static data
                float *wavenumbers_ptr     = (float *) wavenumbers;
                float *spheroidal_ptr      = (float *) spheroidal;
                complex<float> *aterm_ptr = (complex<float> *) aterm;

                // Performance measurements
                double total_runtime_degridder = 0;
                double total_runtime_fft = 0;

                // Start degridder
                #pragma omp target data                            \
                     map(to:wavenumbers_ptr[0:wavenumbers_elements])  \
                     map(to:spheroidal_ptr[0:spheroidal_elements])    \
                     map(to:aterm_ptr[0:aterm_elements])
                {
                    for (unsigned int bl = 0; bl < nr_baselines; bl += jobsize) {
                        // Compute the number of baselines in current job
                        int current_nr_baselines = bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;

                        // Number of elements in batch
                        int uvw_elements          = nr_time * sizeof(UVW)/sizeof(float);
                        int visibilities_elements = nr_time * nr_channels * nr_polarizations;
                        int metadata_elements     = sizeof(Metadata) / sizeof(int);
                        int subgrid_elements      = subgridsize * subgridsize *
                                                    nr_polarizations;

                        // Number of subgrids for all baselines in job
                        auto current_nr_subgrids = plan.get_nr_subgrids(bl, current_nr_baselines);

                        // Pointers to data for current batch
                        float *uvw_ptr = (float *) uvw + bl * uvw_elements;
                        complex<float> *visibilities_ptr = (complex<float>*) visibilities + bl * visibilities_elements;
                        complex<float> *subgrids_ptr     = (complex<float>*) subgrids + subgrid_elements * plan.get_subgrid_offset(bl);
                        int *metadata_ptr     = (int *) plan.get_metadata_ptr(bl);

                        // Power measurement
                        PowerSensor::State powerStates[3];
                        #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                        powerStates[0] = powerSensor->read();
                        #endif

                        // Performance measurement
                        double runtime_degridder, runtime_fft;

                        #pragma omp target \
                            map(tofrom:subgrids_ptr[0:(current_nr_subgrids * subgrid_elements)])
                        {
                            runtime_fft = -omp_get_wtime();

                            kernel::knc::fft(
                                subgridsize,
                                current_nr_subgrids,
                                subgrids_ptr,
                                -1,
                                nr_polarizations);

                            runtime_fft += omp_get_wtime();
                        }

                        #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                        powerStates[1] = powerSensor->read();
                        #endif

                        #pragma omp target \
                            map(to:uvw_ptr[0:(current_nr_baselines * uvw_elements)]) \
                            map(from:visibilities_ptr[0:(current_nr_baselines * visibilities_elements)]) \
                            map(to:subgrids_ptr[0:(current_nr_subgrids * subgrid_elements)]) \
                            map(to:metadata_ptr[0:(current_nr_subgrids * metadata_elements)])
                        {
                            runtime_degridder = -omp_get_wtime();

                            kernel::knc::degridder(
                                current_nr_subgrids,
                                w_offset,
                                uvw_ptr,
                                wavenumbers_ptr,
                                visibilities_ptr,
                                spheroidal_ptr,
                                aterm_ptr,
                                metadata_ptr,
                                subgrids_ptr,
                                nr_stations,
                                nr_time,
                                nr_timeslots,
                                nr_channels,
                                gridsize,
                                subgridsize,
                                imagesize,
                                nr_polarizations);

                            runtime_degridder += omp_get_wtime();
                        }
                        printf("end degridder\n");

                        #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                        powerStates[2] = powerSensor->read();
                        total_runtime_fft += runtime_fft;
                        total_runtime_degridder += runtime_degridder;
                        #endif

                        #if defined(REPORT_VERBOSE)
                        auxiliary::report("fft", runtime_fft,
                            kernel::flops_fft(mParams, subgridsize, current_nr_subgrids),
                            kernel::bytes_fft(mParams, subgridsize, current_nr_subgrids),
                            PowerSensor::Watt(powerStates[0], powerStates[1]) );
                        auxiliary::report("degridder", runtime_degridder,
                            kernel::flops_degridder(mParams, current_nr_baselines, current_nr_subgrids),
                            kernel::bytes_degridder(mParams, current_nr_baselines, current_nr_subgrids),
                            PowerSensor::Watt(powerStates[1], powerStates[2]) );
                        #endif
                    } // end for bl

                } // end pragma omp target data

                #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
                clog << endl;
                auto nr_subgrids = plan.get_nr_subgrids();
                uint64_t total_flops_degridder = kernel::flops_degridder(mParams, nr_baselines, nr_subgrids);
                uint64_t total_bytes_degridder = kernel::bytes_degridder(mParams, nr_baselines, nr_subgrids);
                uint64_t total_flops_fft       = kernel::flops_fft(mParams, subgridsize, nr_subgrids);
                uint64_t total_bytes_fft       = kernel::bytes_fft(mParams, subgridsize, nr_subgrids);
                auxiliary::report("|degridder", total_runtime_degridder, total_flops_degridder, total_bytes_degridder);
                auxiliary::report("|fft", total_runtime_fft, total_flops_fft, total_bytes_fft);
                clog << endl;
                #endif
            }

        } // namespace hybrid
    } // namespace proxy
} // namespace idg





// C interface:
// Rationale: calling the code from C code and Fortran easier,
// and bases to create interface to scripting languages such as
// Python, Julia, Matlab, ...
extern "C" {
    typedef idg::proxy::hybrid::KNC KNC_Offload;

    KNC_Offload* KNC_Offload_init(
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

        return new KNC_Offload(P);
    }

    void KNC_Offload_grid(
        KNC_Offload* p,
        void *visibilities,
        void *uvw,
        void *wavenumbers,
        void *baselines,
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
            (const int*) baselines,
            (std::complex<float>*) grid,
            w_offset,
            kernel_size,
            (const std::complex<float>*) aterm,
            (const int*) aterm_offsets,
            (const float*) spheroidal);
    }

    void KNC_Offload_degrid(
        KNC_Offload* p,
        void *visibilities,
        void *uvw,
        void *wavenumbers,
        void *baselines,
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
            (const int*) baselines,
            (const std::complex<float>*) grid,
            w_offset,
            kernel_size,
            (const std::complex<float>*) aterm,
            (const int*) aterm_offsets,
            (const float*) spheroidal);
    }

    void KNC_Offload_transform(KNC_Offload* p,
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

    void KNC_Offload_destroy(KNC_Offload* p) {
       delete p;
    }

}  // end extern "C"
