#include <iostream>
#include <cmath>

#include "idg.h"
#include "idg-utility.h"  // Data init routines
#include "visualize.h" // HACK

using namespace std;

const int NR_STATIONS = 8;
const int NR_CHANNELS = 8;
const int NR_BASELINES = ((NR_STATIONS - 1)*NR_STATIONS) / 2;
const int NR_TIME = 4096;
const int NR_TIMESLOTS = 1;
const int GRIDSIZE = 1024;
const int SUBGRIDSIZE = 32;
const float IMAGESIZE = 0.3f;
const int NR_POLARIZATIONS = 4;
const float TIME_INTEGRATION = 10.0f;

// const int NR_STATIONS = 2;
// const int NR_CHANNELS = 1;
// const int NR_BASELINES = ((NR_STATIONS - 1)*NR_STATIONS) / 2;
// const int NR_TIME = 10;
// const int NR_TIMESLOTS = 1;
// const int GRIDSIZE = 512;
// const int SUBGRIDSIZE = 32;
// const float IMAGESIZE = 0.3f;
// const int NR_POLARIZATIONS = 4;
// const float TIME_INTEGRATION = 10.0f;

// TODO: move to utility
vector<idg::Measurement<double,complex<float>>> get_example_measurements()
{
    vector<idg::Measurement<double,complex<float>>> result;
    result.reserve(NR_BASELINES*NR_TIME);

    // Allocate raw memory
    auto size_visibilities = 1ULL * NR_BASELINES*NR_TIME*NR_CHANNELS*NR_POLARIZATIONS;
    auto size_uvw          = 1ULL * NR_BASELINES*NR_TIME*3;
    auto size_baselines    = 1ULL * NR_BASELINES*2;

    auto visibilities      = new complex<float>[size_visibilities];
    auto uvw_float         = new float[size_uvw];
    auto uvw_double        = new double[size_uvw];
    auto baselines         = new int[size_baselines];

    idg::init_example_visibilities(visibilities, NR_BASELINES,
                                   NR_TIME, NR_CHANNELS,
                                   NR_POLARIZATIONS);
    idg::init_example_uvw(uvw_float, NR_STATIONS, NR_BASELINES, NR_TIME,
                          TIME_INTEGRATION);
    idg::init_example_baselines(baselines, NR_STATIONS, NR_BASELINES);

    for (auto k = 0; k < size_uvw; ++k) {
        uvw_double[k] = uvw_float[k];
    }

    auto row  = 0;
    for (auto time = 0; time < NR_TIME; ++time) {

        for (auto bl = 0; bl < NR_BASELINES; ++bl) {

            auto antenna1 = baselines[bl*2];
            auto antenna2 = baselines[bl*2 + 1];

            auto u = uvw_double[bl*NR_TIME*3 + time*3 + 0];
            auto v = uvw_double[bl*NR_TIME*3 + time*3 + 1];
            auto w = uvw_double[bl*NR_TIME*3 + time*3 + 2];

            idg::UVWCoordinate<double> uvw = {u, v, w};
            idg::VisibilityGroup<complex<float>> vis(NR_CHANNELS);
            for (auto chan = 0; chan < NR_CHANNELS; ++chan) {
                size_t index = bl*NR_TIME*NR_CHANNELS*NR_POLARIZATIONS
                               + time*NR_CHANNELS*NR_POLARIZATIONS
                               + chan*NR_POLARIZATIONS;
                vis[chan] = {visibilities[index + 0], visibilities[index + 1],
                             visibilities[index + 2], visibilities[index + 3]};
            }

            idg::Measurement<double,complex<float>> M(row, time, antenna1, antenna2, uvw, vis);
            result.push_back(M);

            ++row;
        }
    }

    delete [] visibilities;
    delete [] uvw_float;
    delete [] uvw_double;
    delete [] baselines;

    return result;
}


// TODO: move to utility
vector<double> get_example_frequencies()
{
    vector<double> frequencies;
    frequencies.reserve(NR_CHANNELS);

    auto wavenumbers = new float[NR_CHANNELS];
    idg::init_example_wavenumbers(wavenumbers, NR_CHANNELS);

    for (auto chan = 0; chan < NR_CHANNELS; ++chan) {
        frequencies.push_back( 299792458.0 * wavenumbers[chan] / (2 * M_PI) );
    }

    delete [] wavenumbers;

    return frequencies;
}



int main(int argc, char *argv[])
{
    // Set params
    idg::Parameters params;
    params.set_nr_stations(NR_STATIONS);
    params.set_nr_channels(NR_CHANNELS);
    params.set_nr_time(NR_TIME);
    params.set_nr_timeslots(NR_TIMESLOTS);
    params.set_imagesize(IMAGESIZE);
    params.set_grid_size(GRIDSIZE);

    float w_offset = 0;
    int kernel_size = (SUBGRIDSIZE / 4) + 1;

    clog << params;
    clog << endl;

    // Get measurement data
    clog << ">>> Initialize data structures" << endl;
    auto measurements = get_example_measurements();
    auto frequencies  = get_example_frequencies();

    idg::Grid2D<float> spheroidal_float(SUBGRIDSIZE, SUBGRIDSIZE);
    idg::init_identity_spheroidal(spheroidal_float.data(), SUBGRIDSIZE);

    idg::Grid2D<double> spheroidal_double(SUBGRIDSIZE, SUBGRIDSIZE);
    for (auto y = 0; y < SUBGRIDSIZE; ++y)
        for (auto x = 0; x < SUBGRIDSIZE; ++x)
            spheroidal_double(y, x) = spheroidal_float(y, x);
    // TODO: write move constructor and copy method such that above becomes
    // idg::Grid2D<double> spheroidal_double = spheroidail_float.copy() or so

    idg::Grid3D<complex<double>> image(NR_POLARIZATIONS, GRIDSIZE, GRIDSIZE);
    int y_offset = 300;
    int x_offset = 300;
    image(0, GRIDSIZE/2 + y_offset, GRIDSIZE/2 + x_offset) = 1;
    image(1, GRIDSIZE/2 + y_offset, GRIDSIZE/2 + x_offset) = 1;
    image(2, GRIDSIZE/2 + y_offset, GRIDSIZE/2 + x_offset) = 1;
    image(3, GRIDSIZE/2 + y_offset, GRIDSIZE/2 + x_offset) = 1;

    idg::Grid3D<complex<double>> grid(NR_POLARIZATIONS, GRIDSIZE, GRIDSIZE);
    clog << endl;

    auto bufferSize = NR_TIME / 4;

    #if defined(DEBUG)
    cout << "BUFFER SIZE = " << bufferSize << endl;
    #endif

    auto size_visibilities = 1ULL * NR_CHANNELS*NR_POLARIZATIONS;
    auto visibilities      = new complex<float>[size_visibilities];

    // /////////////////////////////////////////////////////////////////////

    idg::DegridderPlan degridder(idg::Type::CPU_OPTIMIZED, bufferSize);
    degridder.set_stations(NR_STATIONS);
    degridder.set_frequencies(NR_CHANNELS, frequencies.data());
    degridder.set_image(NR_POLARIZATIONS, GRIDSIZE, GRIDSIZE, image.data());
    degridder.set_spheroidal(SUBGRIDSIZE, spheroidal_double.data());
    degridder.set_image_size(IMAGESIZE);
    degridder.set_w_kernel(kernel_size);
    degridder.internal_set_subgrid_size(SUBGRIDSIZE);
    degridder.bake();

    idg::GridderPlan gridder(idg::Type::CPU_OPTIMIZED, bufferSize);
    gridder.set_stations(NR_STATIONS);
    gridder.set_frequencies(NR_CHANNELS, frequencies.data());
    gridder.set_grid(NR_POLARIZATIONS, GRIDSIZE, GRIDSIZE, grid.data());
    gridder.set_spheroidal(SUBGRIDSIZE, spheroidal_double.data());
    gridder.set_image_size(IMAGESIZE);
    gridder.set_w_kernel(kernel_size);
    gridder.internal_set_subgrid_size(SUBGRIDSIZE);
    gridder.bake();


    bool is_buffer_full = false;
    auto nr_rows = measurements.size();

    #if defined(DEBUG)
    cout << "PROCESSING: " << nr_rows << " rows" << endl;
    #endif

    // (1) Predict visibilities: tranform + degrid + write visibilities

    // Since we did initialize with the image, we need to transfor the grid
    degridder.image_to_fourier();

    for (auto row = 0; row < nr_rows; ++row) {

        is_buffer_full = degridder.request_visibilities(
            row,
            measurements[row].get_time_index(),
            measurements[row].get_antenna1(),
            measurements[row].get_antenna2(),
            measurements[row].get_uvw_ptr());

        if (is_buffer_full || row == nr_rows-1) {

            // Compute the requested visibilities
            auto available_row_ids = degridder.compute();

            // Read all available visibilities
            for (auto& r : available_row_ids) {
                degridder.read_visibilities(r, visibilities);
                // hack to write predicted visibilities
                copy(visibilities, visibilities + NR_CHANNELS * NR_POLARIZATIONS,
                     (complex<float>*) measurements[r].get_visibilities_ptr());
            }

            // Signal that we can start requesting again
            degridder.finished_reading();

            // Push failed request again (always fits) before continue the loop
            degridder.request_visibilities(
                row,
                measurements[row].get_time_index(),
                measurements[row].get_antenna1(),
                measurements[row].get_antenna2(),
                measurements[row].get_uvw_ptr());
        }

    }

    // (2) Image predicted visibilities: grid + transform
    for (auto row = 0; row < nr_rows; ++row) {

        gridder.grid_visibilities(
            measurements[row].get_time_index(),
            measurements[row].get_antenna1(),
            measurements[row].get_antenna2(),
            measurements[row].get_uvw_ptr(),
            measurements[row].get_visibilities_ptr());

    }

    gridder.finished();

    idg::NamedWindow fig1("grid");
    fig1.display_matrix(GRIDSIZE, GRIDSIZE, grid.data(), "log", "jet");

    gridder.transform_grid();

    idg::NamedWindow fig2("image");
    fig2.display_matrix(GRIDSIZE, GRIDSIZE, grid.data(), "abs", "hot");

    return 0;
}
