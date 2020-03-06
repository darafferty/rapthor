#include <iostream>
#include <iomanip>
#include <cstdlib> // size_t
#include <complex>
#include <tuple>
#include <typeinfo>
#include <vector>
#include <algorithm> // max_element
#include <numeric> // accumulate
#include <mutex>

#include "idg-cpu.h"
#include "idg-util.h"  // Data init routines

/*
 * Visibilities initializion
 *   0, use simulated visibilities for some point sources near the centre of the image
 *   1, use fixed (dummy) visibilities to speed-up initialization of the data
 */
#define USE_DUMMY_VISIBILITIES 1

using namespace std;

std::tuple<int, int, int, int, int, int, int, int, int, int, int, float>read_parameters() {
    const unsigned int DEFAULT_NR_STATIONS  = 52; // all LOFAR LBA stations
    const unsigned int DEFAULT_NR_CHANNELS  = 16*4; // 16 channels, 4 subbands
    const unsigned int DEFAULT_NR_TIMESTEPS = (3600 * 4); // 4 hours of observation
    const unsigned int DEFAULT_NR_TIMESLOTS = DEFAULT_NR_TIMESTEPS / (60 * 30); // update every 30 minutes
    const unsigned int DEFAULT_GRIDSIZE     = 4096;
    const unsigned int DEFAULT_SUBGRIDSIZE  = 32;
    const unsigned int DEFAULT_NR_CYCLES    = 1;
    const float        DEFAULT_GRID_PADDING = 1.0;

    char *cstr_nr_stations = getenv("NR_STATIONS");
    auto nr_stations = cstr_nr_stations ? atoi(cstr_nr_stations): DEFAULT_NR_STATIONS;

    char *cstr_nr_channels = getenv("NR_CHANNELS");
    auto nr_channels = cstr_nr_channels ? atoi(cstr_nr_channels) : DEFAULT_NR_CHANNELS;

    char *cstr_nr_timesteps = getenv("NR_TIMESTEPS");
    auto nr_timesteps = cstr_nr_timesteps ? atoi(cstr_nr_timesteps) : DEFAULT_NR_TIMESTEPS;

    char *cstr_total_nr_stations = getenv("TOTAL_NR_STATIONS");
    auto total_nr_stations = cstr_total_nr_stations ? atoi(cstr_total_nr_stations): nr_stations;

    char *cstr_total_nr_channels = getenv("TOTAL_NR_CHANNELS");
    auto total_nr_channels = cstr_total_nr_channels ? atoi(cstr_total_nr_channels) : nr_channels;

    char *cstr_total_nr_timesteps = getenv("TOTAL_NR_TIMESTEPS");
    auto total_nr_timesteps = cstr_total_nr_timesteps ? atoi(cstr_total_nr_timesteps) : nr_timesteps;

    char *cstr_nr_timeslots = getenv("NR_TIMESLOTS");
    auto nr_timeslots = cstr_nr_timeslots ? atoi(cstr_nr_timeslots) : DEFAULT_NR_TIMESLOTS;

    char *cstr_grid_size = getenv("GRIDSIZE");
    auto grid_size = cstr_grid_size ? atoi(cstr_grid_size) : DEFAULT_GRIDSIZE;

    char *cstr_subgrid_size = getenv("SUBGRIDSIZE");
    auto subgrid_size = cstr_subgrid_size ? atoi(cstr_subgrid_size) : DEFAULT_SUBGRIDSIZE;

    char *cstr_kernel_size = getenv("KERNELSIZE");
    auto kernel_size = cstr_kernel_size ? atoi(cstr_kernel_size) : (subgrid_size / 4) + 1;

    char *cstr_nr_cycles = getenv("NR_CYCLES");
    auto nr_cycles = cstr_nr_cycles ? atoi(cstr_nr_cycles) : DEFAULT_NR_CYCLES;

    char *cstr_grid_padding = getenv("GRID_PADDING");
    auto grid_padding = cstr_grid_padding ? atof(cstr_grid_padding) : DEFAULT_GRID_PADDING;

    return std::make_tuple(
        total_nr_stations, total_nr_channels, total_nr_timesteps,
        nr_stations, nr_channels, nr_timesteps, nr_timeslots,
        grid_size, subgrid_size, kernel_size,
        nr_cycles, grid_padding);
}

void print_parameters(
    unsigned int total_nr_stations,
    unsigned int total_nr_channels,
    unsigned int total_nr_timesteps,
    unsigned int nr_stations,
    unsigned int nr_channels,
    unsigned int nr_timesteps,
    unsigned int nr_timeslots,
    float image_size,
    unsigned int grid_size,
    unsigned int subgrid_size,
    unsigned int kernel_size,
    float grid_padding
) {
    const int fw1 = 30;
    const int fw2 = 10;
    ostream &os = clog;

    os << "-----------" << endl;
    os << "PARAMETERS:" << endl;

    os << setw(fw1) << left << "Total number of stations" << "== "
       << setw(fw2) << right << total_nr_stations << endl;

    os << setw(fw1) << left << "Total number of channels" << "== "
       << setw(fw2) << right << total_nr_channels << endl;

    os << setw(fw1) << left << "Total number of timesteps" << "== "
       << setw(fw2) << right << total_nr_timesteps << endl;

    os << setw(fw1) << left << "Number of stations" << "== "
       << setw(fw2) << right << nr_stations << endl;

    os << setw(fw1) << left << "Number of channels" << "== "
       << setw(fw2) << right << nr_channels << endl;

    os << setw(fw1) << left << "Number of timesteps" << "== "
       << setw(fw2) << right << nr_timesteps << endl;

    os << setw(fw1) << left << "Number of timeslots" << "== "
       << setw(fw2) << right << nr_timeslots << endl;

    os << setw(fw1) << left << "Imagesize" << "== "
       << setw(fw2) << right << image_size  << endl;

    os << setw(fw1) << left << "Grid size" << "== "
       << setw(fw2) << right << grid_size << endl;

    os << setw(fw1) << left << "Subgrid size" << "== "
       << setw(fw2) << right << subgrid_size << endl;

    os << setw(fw1) << left << "Kernel size" << "== "
       << setw(fw2) << right << kernel_size << endl;

    os << setw(fw1) << left << "Grid padding" << "== "
       << setw(fw2) << right << grid_padding << endl;

    os << "-----------" << endl;
}

void run()
{
    idg::auxiliary::print_version();

    // Constants
    unsigned int nr_w_layers = 1;
    unsigned int nr_correlations = 4;
    float w_offset = 0;
    unsigned int total_nr_stations;
    unsigned int total_nr_timesteps;
    unsigned int total_nr_channels;
    unsigned int nr_stations;
    unsigned int nr_channels;
    unsigned int nr_timesteps;
    unsigned int nr_timeslots;
    float integration_time = 1.0;
    unsigned int grid_size;
    unsigned int subgrid_size;
    unsigned int kernel_size;
    unsigned int nr_cycles;
    float grid_padding;

    // Read parameters from environment
    std::tie(
        total_nr_stations, total_nr_channels, total_nr_timesteps,
        nr_stations, nr_channels, nr_timesteps, nr_timeslots,
        grid_size, subgrid_size, kernel_size,
        nr_cycles, grid_padding) = read_parameters();
    unsigned int nr_baselines = (nr_stations * (nr_stations - 1)) / 2;

    // Initialize Data object
    clog << ">>> Initialize data" << endl;
    idg::Data data;

    // Determine the max baseline length for given grid_size
    auto max_uv = data.compute_max_uv(grid_padding * grid_size);

    // Select only baselines up to max_uv meters long
    data.limit_max_baseline_length(max_uv);

    // Restrict the number of baselines to nr_baselines
    data.limit_nr_baselines(nr_baselines);

    // Print data info
    data.print_info();

    // Get remaining parameters
    nr_baselines = data.get_nr_baselines();
    float image_size = data.compute_image_size(grid_padding * grid_size);
    float cell_size = image_size / grid_size;

    // Print parameters
    print_parameters(
        total_nr_stations, total_nr_channels, total_nr_timesteps,
        nr_stations, nr_channels, nr_timesteps, nr_timeslots,
        image_size, grid_size, subgrid_size, kernel_size, grid_padding);

    // Warn for unrealistic number of timesteps
    float observation_length = (total_nr_timesteps * integration_time) / 3600;
    if (observation_length > 12) {
        clog << "Observation length of: " << observation_length
             << " hours (> 12) selected!" << endl;
    }

    // Initialize proxy
    clog << ">>> Initialize proxy" << endl;
    ProxyType proxy;
    clog << endl;

    // Allocate and initialize static data structures
    clog << ">>> Initialize data structures" << endl;
    #if USE_DUMMY_VISIBILITIES
    idg::Array3D<idg::Visibility<std::complex<float>>> visibilities_ =
        idg::get_dummy_visibilities(proxy, nr_baselines, nr_timesteps, nr_channels);
    #endif
    idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterms =
        idg::get_identity_aterms(proxy, nr_timeslots, nr_stations, subgrid_size, subgrid_size);
    idg::Array1D<unsigned int> aterms_offsets =
        idg::get_example_aterms_offsets(proxy, nr_timeslots, nr_timesteps);
    idg::Array2D<float> spheroidal =
        idg::get_example_spheroidal(proxy, subgrid_size, subgrid_size);
    auto grid =
        proxy.allocate_grid(nr_w_layers, nr_correlations, grid_size, grid_size);
    idg::Array1D<float> shift =
        idg::get_zero_shift();
    idg::Array1D<std::pair<unsigned int,unsigned int>> baselines =
        idg::get_example_baselines(proxy, nr_stations, nr_baselines);
    clog << endl;

    // Benchmark
    vector<double> runtimes_gridding;
    vector<double> runtimes_degridding;
    vector<double> runtimes_fft;
    vector<double> runtimes_imaging;
    unsigned long nr_visibilities = 0;

    // Enable/disable routines
    bool disable_gridding   = getenv("DISABLE_GRIDDING");
    bool disable_degridding = getenv("DISABLE_DEGRIDDING");
    bool disable_fft        = getenv("DISABLE_FFT");

    // Spectral line imaging
    bool simulate_spectral_line = getenv("SPECTRAL_LINE");

    // Vectors for Plan and UVW
    idg::Plan::Options options;
    options.plan_strict = true;
    options.simulate_spectral_line = simulate_spectral_line;
    options.max_nr_timesteps_per_subgrid = 128;
    options.max_nr_channels_per_subgrid = 8;
    omp_set_nested(true);

    // Iterate all cycles
    for (unsigned cycle = 0; cycle < nr_cycles; cycle++) {

        // Set grid
        proxy.set_grid(grid);

        // Iterate all time blocks
        for (unsigned time_offset = 0; time_offset < total_nr_timesteps; time_offset += nr_timesteps) {
            int current_nr_timesteps = total_nr_timesteps - time_offset < nr_timesteps ?
                                   total_nr_timesteps - time_offset : nr_timesteps;

            // Initalize UVW coordiantes
            auto uvw = proxy.allocate_array2d<idg::UVW<float>>(nr_baselines, current_nr_timesteps);
            data.get_uvw(uvw, 0, time_offset, integration_time);

            // Iterate all channel blocks
            for (unsigned channel_offset = 0; channel_offset < total_nr_channels; channel_offset += nr_channels) {
                // Report progress
                clog << ">>>" << endl;
                clog << "time: " << time_offset << "-" << time_offset + nr_timesteps << ", ";
                clog << "channel: " << channel_offset << "-" << channel_offset + nr_channels << endl;
                clog << ">>>" << endl;

                // Initialize frequency data
                idg::Array1D<float> frequencies = proxy.allocate_array1d<float>(nr_channels);
                data.get_frequencies(frequencies, image_size, channel_offset);

                // Initialize visibilities
                #if !USE_DUMMY_VISIBILITIES
                idg::Array3D<idg::Visibility<std::complex<float>>> visibilities_ =
                    idg::get_example_visibilities(proxy, uvw, frequencies, image_size, grid_size);
                #endif


                auto nr_channels_ = simulate_spectral_line ? 1 : nr_channels;
                idg::Array3D<idg::Visibility<std::complex<float>>> visibilities(visibilities_.data(), nr_baselines, current_nr_timesteps, nr_channels_);

                // Create plan
                auto plan = std::unique_ptr<idg::Plan>(new idg::Plan(
                    kernel_size, subgrid_size, grid_size, cell_size,
                    frequencies, uvw, baselines, aterms_offsets, options));

                // Start imaging
                double runtime_imaging = -omp_get_wtime();

                // Run gridding
                clog << ">>> Run gridding" << endl;
                double runtime_gridding = -omp_get_wtime();
                if (!disable_gridding)
                proxy.gridding(
                    *plan, w_offset, shift, cell_size, kernel_size, subgrid_size,
                    frequencies, visibilities, uvw, baselines,
                    *grid, aterms, aterms_offsets, spheroidal);
                runtimes_gridding.push_back(runtime_gridding + omp_get_wtime());
                clog << endl;

                // Run degridding
                clog << ">>> Run degridding" << endl;
                double runtime_degridding = -omp_get_wtime();
                if (!disable_degridding)
                proxy.degridding(
                    *plan, w_offset, shift, cell_size, kernel_size, subgrid_size,
                    frequencies, visibilities, uvw, baselines,
                    *grid, aterms, aterms_offsets, spheroidal);
                runtimes_degridding.push_back(runtime_degridding + omp_get_wtime());
                clog << endl;

                // End imaging
                runtimes_imaging.push_back(runtime_imaging + omp_get_wtime());
            } // end for channel_offset
        } // end for time_offset

        // Run fft
        clog << ">>> Run fft" << endl;
        double runtime_fft = -omp_get_wtime();
        if (!disable_fft)
        for (unsigned w = 0; w < nr_w_layers; w++) {
            idg::Array3D<std::complex<float>> grid_(grid->data(w), nr_correlations, grid_size, grid_size);
            proxy.transform(idg::FourierDomainToImageDomain, grid_);
            proxy.transform(idg::ImageDomainToFourierDomain, grid_);
        }
        runtimes_fft.push_back(runtime_fft + omp_get_wtime());
        clog << endl;

        // Only after a call to get_grid(), the grid can be used outside of the proxy
        proxy.get_grid();
    } // end for i (nr_cycles)

    // Compute maximum runtime
    double max_runtime_gridding   = *max_element(runtimes_gridding.begin(), runtimes_gridding.end());
    double max_runtime_degridding = *max_element(runtimes_degridding.begin(), runtimes_degridding.end());
    double max_runtime_fft        = *max_element(runtimes_fft.begin(), runtimes_fft.end());
    double max_runtime_imaging    = *max_element(runtimes_imaging.begin(), runtimes_imaging.end());

    // Compute total runtime
    double runtime_gridding       = accumulate(runtimes_gridding.begin(), runtimes_gridding.end(), 0.0);
    double runtime_degridding     = accumulate(runtimes_degridding.begin(), runtimes_degridding.end(), 0.0);
    double runtime_fft            = accumulate(runtimes_fft.begin(), runtimes_fft.end(), 0.0);
    double runtime_imaging        = accumulate(runtimes_imaging.begin(), runtimes_imaging.end(), 0.0);

    // Ignore slowest run
    if (nr_cycles > 1) {
        runtime_gridding   -= max_runtime_gridding;
        runtime_degridding -= max_runtime_degridding;
        runtime_fft        -= max_runtime_fft;
        runtime_imaging    -= max_runtime_imaging;
        nr_cycles          -= 1;
    }

    // Compute runtime for one cycle
    runtime_gridding   /= nr_cycles;
    runtime_degridding /= nr_cycles;
    runtime_imaging    /= nr_cycles;
    runtime_imaging    /= nr_cycles;

    // Report runtime
    idg::report("gridding", runtime_gridding);
    idg::report("degridding", runtime_degridding);
    idg::report("fft", runtime_fft);
    idg::report("imaging", runtime_imaging);

    // Report throughput
    idg::report_visibilities("gridding", runtime_gridding, nr_visibilities);
    idg::report_visibilities("degridding", runtime_degridding, nr_visibilities);
    idg::report_visibilities("imaging", runtime_imaging, nr_visibilities);
}
