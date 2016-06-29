#include <iostream>
#include <stdexcept>
#include <memory>
#include <vector>

#include "casacore/ms/MeasurementSets.h"
#include "casacore/casa/Arrays/ArrayMath.h"
#include "idg.h"
#include "visualize.h"

using namespace std;


int run_demo(string ms_name,
             int percentage)
{
    #if defined(DEBUG)
    cout << __func__ << "(" << ms_name << ", " << percentage << ")" << endl;
    #endif

    // Open measurement set
    casacore::MeasurementSet ms(ms_name);
    casacore::ROMSColumns msCols(ms);
    int nr_antennas  = msCols.antenna().nrow();
    int nr_channels  = msCols.spectralWindow().numChan()(0);
    auto frequencies = msCols.spectralWindow().chanFreq()(0);
    // int nr_data_rows = msCols.data().nrow();

    // alternatively:
    casacore::Table table(ms_name);
    casacore::ROScalarColumn<casacore::Double> timeCol(table, "TIME");
    casacore::ROScalarColumn<casacore::Int> antenna1Col(table, "ANTENNA1");
    casacore::ROScalarColumn<casacore::Int> antenna2Col(table, "ANTENNA2");
    casacore::ROArrayColumn<casacore::Double> uvwCol(table, "UVW");
    casacore::ROArrayColumn<casacore::Complex> dataCol(table, "DATA");
    casacore::ROArrayColumn<casacore::Bool> flagCol(table, "FLAG");

    // Parameters
    //casacore::ROScalarColumn<casacore::Double> freqCol(table, "WEIGHT_SPECTRUM");
    //??? baselineCol(table, "BASELINE");
    //cout << freqCol(0) << endl;
    //exit(0);

    int nr_data_rows = table.nrow();

    #if defined(DEBUG)
    cout << "NR_DATA_ROWS = " << nr_data_rows << endl;
    cout << "NR_ANTENNAS  = " << nr_antennas  << endl;
    cout << "NR_CHANNELS  = " << nr_channels  << endl;
    #endif

    // Initialize proxy
    int buffer_timesteps = 256;
    int nr_timeslots     = 1;
    int nr_polarizations = 4;
    int grid_size        = 1024;
    float cell_size      = 0.05/grid_size;
    int subgrid_size     = 32;
    int kernel_size      = subgrid_size/2;

    auto size_grid  = 1ULL * nr_polarizations * grid_size * grid_size;
    unique_ptr<complex<double>> grid(new complex<double>[size_grid]);

    idg::GridderPlan gridder(idg::Type::CPU_OPTIMIZED, buffer_timesteps);
    gridder.set_stations(nr_antennas);
    gridder.set_frequencies(nr_channels, frequencies.data());
    gridder.set_grid(nr_polarizations, grid_size, grid_size, grid.get());
    gridder.set_cell_size(cell_size, cell_size);
    gridder.set_w_kernel(kernel_size);
    gridder.internal_set_subgrid_size(subgrid_size);
    gridder.bake();

    // loop over data rows and grid visibilities
    int    timeIndex = -1;
    double time_previous = 1;
    auto visibilities_copy = new complex<float>[nr_channels * nr_polarizations];
    auto flags_copy = new bool[nr_channels * nr_polarizations];
    for (auto row = 0; row < nr_data_rows * (percentage/100.); row++) {

        auto time         = timeCol.get(row);
        auto antenna1     = antenna1Col.get(row);
        auto antenna2     = antenna2Col.get(row);
        // auto visibilities = dataCol.get(row).data();
        // auto uvw          = uvwCol.get(row).data();

        double uvw_copy[3];
        int k = 0;
        for (auto &x : uvwCol.get(row)) {
            uvw_copy[k] = x;
            k++;
        }

        k = 0;
        for (auto &x : flagCol.get(row)) {
            flags_copy[k] = x;
            k++;
        }

        k = 0;
        for (auto &x : dataCol.get(row)) {
            if (!flags_copy[k]) visibilities_copy[k] = x;
            else visibilities_copy[k] = 0;
            k++;
        }

        if (fabs(time - time_previous)) timeIndex++;

        // TODO: apply (1 - flags) * visibilities to set flagged data to zero
        // auto tmp_visibilities = dataCol(row);
        // auto tmp_flags        = flagCol(row);
        // cout << tmp_visibilities << endl;
        // cout << tmp_flags << endl;


        // #if defined(DEBUG)
        // cout << "time =" << time
        //      << ", timeIndex = " << timeIndex
        //      << ", antenna1 = " << antenna1
        //      << ", antenna2 = " << antenna2
        //      << ", uvw = " << uvwCol(row) << endl;
        // #endif

        // cout << msCols.data()(row) << endl;

        gridder.grid_visibilities(
            timeIndex,
            antenna1,
            antenna2,
            uvw_copy,
            visibilities_copy);

        time_previous = time;
    }

    gridder.flush(); // makes sure buffer is empty at the end
    delete [] visibilities_copy;
    delete [] flags_copy;

    idg::display_complex_matrix(grid_size, grid_size, grid.get(), "log");

    gridder.transform_grid();

    idg::display_complex_matrix(grid_size, grid_size, grid.get(), "abs");

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
