#include <iostream>
#include <stdexcept>
#include <memory>
#include <chrono>

#include "MS-hdf5.h"
#include "idg.h"
#include "visualize.h"

using namespace std;


int run_demo(string ms_name,
             int percentage)
{
    #if defined(DEBUG)
    cout << __func__ << "(" << ms_name << ", " << percentage << ")" << endl;
    #endif

    // Plotting
    idg::NamedWindow fig("Gridded visibilities");

    // Open measurement set
    idg::MS_hdf5 hdf5_file(ms_name, H5F_ACC_RDONLY);

    // Get attributes
    unsigned int nr_antennas     = hdf5_file.get_nr_antennas();
    unsigned int nr_baselines    = hdf5_file.get_nr_baselines();
    unsigned int nr_timesteps    = hdf5_file.get_nr_timesteps();
    unsigned int nr_channels     = hdf5_file.get_nr_channels();
    unsigned int nr_correlations = hdf5_file.get_nr_correlations();

    // Read data
    auto frequencies = hdf5_file.read_frequencies();
    auto antenna1    = hdf5_file.read_antenna1();
    auto antenna2    = hdf5_file.read_antenna2();

    #if defined(DEBUG)
    cout << endl;
    cout << "PARAMETERS READ FROM MS" << endl;
    cout << "-----------------------" << endl;
    cout << "Number of baselines:   " << nr_baselines    << endl;
    cout << "Number of correlation: " << nr_correlations << endl;
    cout << "Number of channels:    " << nr_channels     << endl;
    cout << "Number of antennas:    " << nr_antennas     << endl;
    cout << endl;
    #endif

    // Set parameters from proxy
    int buffer_timesteps = 256;
    int nr_timeslots     = 1;
    int nr_polarizations = nr_correlations;
    int grid_size        = 1024;
    float cell_size      = 0.05/grid_size;
    int subgrid_size     = 32;
    int kernel_size      = subgrid_size/2;

    #if defined(DEBUG)
    cout << "PARAMETERS FOR PROXY" << endl;
    cout << "-----------------------" << endl;
    cout << "Buffer timesteps:      " << buffer_timesteps << endl;
    cout << "Cell size:             " << cell_size        << endl;
    cout << "Grid size:             " << grid_size        << endl;
    #endif

    // Allocate memory
    idg::Grid3D<complex<double>> grid(nr_polarizations, grid_size, grid_size);

    // Initialize proxy
    // TODO: nicer constructor that does all of this at once,
    // only taking the necessary arguments for compilation,
    // the rest can be infered from the argments using IDG data types
    idg::GridderPlan gridder(idg::Type::CPU_OPTIMIZED, buffer_timesteps);
    gridder.set_stations(nr_antennas);
    gridder.set_frequencies(nr_channels, frequencies.data());
    gridder.set_grid(nr_polarizations, grid_size, grid_size, grid.data());
    gridder.set_cell_size(cell_size, cell_size);
    gridder.set_w_kernel(kernel_size);
    gridder.internal_set_subgrid_size(subgrid_size);
    gridder.bake();

    #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
    chrono::high_resolution_clock::time_point t_start_block;
    chrono::high_resolution_clock::time_point t_end_block;
    t_start_block = chrono::high_resolution_clock::now();
    #endif

    int number_timestips = min(nr_timesteps * (double)percentage/100, nr_timesteps);

    // TODO: the data can also be loaded in chunks (more than one timestep at a time)
    // Is this faster/better?
    for (auto t = 0; t < number_timestips; ++t) {

        // TODO: if expensive, do not allocate and initialize arrays every timestep
        auto visibilities    = hdf5_file.read_visibilities(t);
        auto flags           = hdf5_file.read_flags(t);
        auto uvw_coordinates = hdf5_file.read_uvw_coordinates(t);

        // Apply flags
        // TODO: have an api that takes visibilities and flags separately
        // maybe also an optional argument with weights
        idg::apply_flags(visibilities, flags);

        // TODO: nicer way of calling IDG gridder with IDG data types
        for (auto bl = 0; bl < nr_baselines; ++bl) {
            gridder.grid_visibilities(
                t,
                antenna1[bl],
                antenna2[bl],
                (double*) uvw_coordinates.data(0,bl),
                (complex<float>*) visibilities.data(0,bl)
            );
        }

        // The rest of the loop body for timing and plotting purposes only
        if (t % buffer_timesteps == 1) {
             #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
             // display total time of processing the block (without plotting)
             t_end_block = chrono::high_resolution_clock::now();
             auto t_block = chrono::duration_cast<chrono::milliseconds>(
                 t_end_block - t_start_block).count();
             cout << "Runtime total: " << t_block << " ms" << endl;

             // start plot
             auto t_start_plot = chrono::high_resolution_clock::now();
             #endif

             // Display grid (note: equals one as buffer flushed iteration AFTER being full)
             fig.display_matrix(grid_size, grid_size, grid.data(), "log", "jet");

             #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
             auto t_end_plot = chrono::high_resolution_clock::now();
             auto t_plot = chrono::duration_cast<chrono::milliseconds>(
                 t_end_plot - t_start_plot).count();
             cout << "Runtime plot: " << t_plot << " ms" << endl;
             #endif

             #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
             t_start_block = chrono::high_resolution_clock::now();
             #endif
        }

    }

    // Make sure buffer is empty at the end
    gridder.finished();

    #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
    t_end_block = chrono::high_resolution_clock::now();
    auto t_block = chrono::duration_cast<chrono::milliseconds>(
        t_end_block - t_start_block).count();
    cout << "Runtime total: " << t_block << " ms" << endl;

    auto t_start_plot = chrono::high_resolution_clock::now();
    #endif

    // Display results
    fig.display_matrix(grid_size, grid_size, grid.data(), "log", "jet");

    #if defined(REPORT_VERBOSE) || defined(REPORT_TOTAL)
    auto t_end_plot = chrono::high_resolution_clock::now();
    auto t_plot = chrono::duration_cast<chrono::milliseconds>(
    t_end_plot - t_start_plot).count();
    cout << "Runtime plot: " << t_plot << " ms" << endl;
    #endif

    gridder.transform_grid();

    idg::NamedWindow fig2("Sky image");
    fig2.display_matrix(grid_size, grid_size, grid.data(), "abs", "hot", 1000000);

    return 0;
}




int main(int argc, char *argv[])
{
    int info = 0;

    if (argc==1) {
        cerr << "Usage: " << argv[0] << " MeasurementSet [Percentage]" << endl;
        exit(0);
    }
    string ms_name = argv[1];
    int percentage = 100;
    if (argc > 2) {
        percentage = stoi( argv[2] );
    }
    if (percentage < 0 || percentage > 100) {
        percentage = 100;
    }

    cout << "Measurement Set: " << ms_name << endl;
    cout << "Percentage: " << percentage << endl;

    info = run_demo(ms_name, percentage);

    return info;
}
