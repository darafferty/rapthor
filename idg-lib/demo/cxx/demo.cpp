#include <iostream>
#include <stdexcept>
#include <memory>
#include <vector>

#include "casacore/tables/Tables/TableIter.h"
#include "casacore/tables/Tables/ScalarColumn.h"
#include "casacore/tables/Tables/ArrayColumn.h"
#include <casacore/casa/Arrays/Cube.h>

#include "idg.h"
#include "visualize.h"

using namespace std;

// TODO: make 'percentage' parameter work again
int run_demo(string ms_name,
             int percentage)
{
    #if defined(DEBUG)
    cout << __func__ << "(" << ms_name << ", " << percentage << ")" << endl;
    #endif

    // Plotting
    idg::NamedWindow fig("Gridded visibilities");

    // Open measurement set
    casacore::Table ms_table(ms_name);

    // Get iterator for column TIME
    casacore::TableIterator iter(
        ms_table,
        casacore::Block<casacore::String>(1, "TIME"),
        casacore::TableIterator::Ascending,
        casacore::TableIterator::NoSort);

    // Read frequencies
    casacore::Table channel_table(ms_name + "/SPECTRAL_WINDOW");
    casacore::ArrayColumn<casacore::Double> channel_column(channel_table, "CHAN_FREQ");
    casacore::Matrix<casacore::Double> channel_frequencies = channel_column.getColumn();
    int nr_channels = channel_frequencies.nelements();

    // Read antenna IDs
    casacore::Table antenna_table(ms_name + "/ANTENNA");

    casacore::ScalarColumn<casacore::Int> antenna1_column(iter.table(), "ANTENNA1");
    casacore::ScalarColumn<casacore::Int> antenna2_column(iter.table(), "ANTENNA2");
    casacore::Vector<casacore::Int> ant1 = antenna1_column.getColumn();
    casacore::Vector<casacore::Int> ant2 = antenna2_column.getColumn();

    // Get parameters from MS
    int nr_baselines    = iter.table().nrow();
    int nr_correlations = 4; // Number of correlations: xx, xy, yx, yy
    int nr_antennas     = antenna_table.nrow();

    #if defined(DEBUG)
    cout << "PARAMETERS READ FROM MS" << endl;
    cout << "Number of baselines:   " << nr_baselines    << endl;
    cout << "Number of correlation: " << nr_correlations << endl;
    cout << "Number of channels:    " << nr_channels     << endl;
    cout << "Number of antennas:    " << nr_antennas     << endl;
    cout << endl;
    #endif

    // Set parameters fro proxy
    int buffer_timesteps = 256;
    int nr_timeslots     = 1;
    int nr_polarizations = nr_correlations;
    int grid_size        = 1024;
    float cell_size      = 0.05/grid_size;
    int subgrid_size     = 32;
    int kernel_size      = subgrid_size/2;

    #if defined(DEBUG)
    cout << "PARAMETERS FOR PROXY" << endl;
    cout << "Buffer timesteps:      " << buffer_timesteps << endl;
    cout << "Cell size:             " << cell_size        << endl;
    cout << "Grid size:             " << grid_size        << endl;
    #endif

    // Allocate memory
    auto size_grid  = 1ULL * nr_polarizations * grid_size * grid_size;
    unique_ptr<complex<double>> grid(new complex<double>[size_grid]);

    // Initialize proxy
    idg::GridderPlan gridder(idg::Type::CPU_OPTIMIZED, buffer_timesteps);
    gridder.set_stations(nr_antennas);
    gridder.set_frequencies(nr_channels, channel_frequencies.data());
    gridder.set_grid(nr_polarizations, grid_size, grid_size, grid.get());
    gridder.set_cell_size(cell_size, cell_size);
    gridder.set_w_kernel(kernel_size);
    gridder.internal_set_subgrid_size(subgrid_size);
    gridder.bake();

    // Read data
    casacore::Cube<casacore::Complex> data(nr_correlations, nr_channels, nr_baselines);
    casacore::Matrix<casacore::Double> uvw(3, nr_baselines);
    casacore::Cube<casacore::Bool> flag(nr_correlations, nr_channels, nr_baselines);
    unsigned int time_index = 0;

    while (!iter.pastEnd()) {

        casacore::ScalarColumn<double> time_column(iter.table(), "TIME");
        auto timestamp = time_column.getColumn()(0);

        casacore::ArrayColumn<casacore::Complex> data_column(iter.table(), "DATA");
        casacore::ArrayColumn<casacore::Double> uvw_column(iter.table(), "UVW");
        casacore::ArrayColumn<casacore::Bool> flag_column(iter.table(), "FLAG");

        data_column.getColumn(data);
        uvw_column.getColumn(uvw);
        flag_column.getColumn(flag);

        for (auto bl = 0; bl < nr_baselines; ++bl) {

            auto antenna1 = ant1(bl);
            auto antenna2 = ant2(bl);

            double *uvw_ptr  = uvw.data() + bl*3;
            complex<float> *visibilities_ptr  = data.data() + bl*nr_channels*nr_correlations;
            bool *flag_ptr  = flag.data() + bl*nr_channels*nr_correlations;

            // TODO: clean up the lines below
            for (auto k = 0; k < nr_channels*nr_correlations; ++k) {
                if ( *(flag_ptr + k) ) *(visibilities_ptr + k) = 0;
            }

            gridder.grid_visibilities(
                  time_index,
                  antenna1,
                  antenna2,
                  uvw_ptr,
                  visibilities_ptr);
         }

        // Display grid (note: equals one as buffer flushed iteration AFTER being full)
        if (time_index % buffer_timesteps == 1) {
             fig.display_matrix(grid_size, grid_size, grid.get(), "log", "jet");
        }

        time_index++;
        iter.next();
    }

    // Make sure buffer is empty at the end
    gridder.finished();

    // Display results
    fig.display_matrix(grid_size, grid_size, grid.get(), "log", "jet");

    gridder.transform_grid();

    idg::NamedWindow fig2("Sky image");
    fig2.display_matrix(grid_size, grid_size, grid.get(), "abs", "hot", 1000000);

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
