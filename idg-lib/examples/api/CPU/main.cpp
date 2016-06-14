#include <iostream>
#include <cmath>

#include "idg.h"
#include "idg-utility.h"  // Data init routines

using namespace std;


int main(int argc, char *argv[])
{
    // Set constants explicitly in the parameters parameter
    std::clog << ">>> Configuration"  << std::endl;
    idg::Parameters params;
    // Read the following from ENV:
    // NR_STATIONS, NR_CHANNELS, NR_TIMESTEPS, NR_TIMESLOTS, IMAGESIZE,
    // GRIDSIZE
    // if non-default jobsize wanted, set also JOBSIZE, etc.
    params.set_from_env();

    // retrieve constants for memory allocation
    int nr_stations      = params.get_nr_stations();
    int nr_baselines     = params.get_nr_baselines();
    int nr_time          = params.get_nr_time();
    int nr_timeslots     = params.get_nr_timeslots();
    int nr_channels      = params.get_nr_channels();
    int gridsize         = params.get_grid_size();
    int subgridsize      = params.get_subgrid_size();
    float imagesize      = params.get_imagesize();
    int nr_polarizations = params.get_nr_polarizations();

    float w_offset = 0;
    int kernel_size = (subgridsize / 4) + 1;

    // Print configuration
    std::clog << params;
    std::clog << std::endl;

    // Allocate and initialize data structures
    std::clog << ">>> Initialize data structures" << std::endl;

    auto size_visibilities = 1ULL * nr_channels*nr_polarizations;
    auto size_uvw          = 1ULL * nr_baselines*nr_time*3;
    auto size_wavenumbers  = 1ULL * nr_channels;
    auto size_aterm        = 1ULL * nr_timeslots*nr_stations*
                             subgridsize*subgridsize*nr_polarizations;
    auto size_spheroidal   = 1ULL * subgridsize*subgridsize;
    auto size_grid         = 1ULL * nr_polarizations*gridsize*gridsize;
    auto size_baselines    = 1ULL * nr_baselines*2;

    auto visibilities      = new std::complex<float>[size_visibilities];
    auto uvw               = new float[size_uvw];
    auto uvw_double        = new double[size_uvw];
    auto wavenumbers       = new float[size_wavenumbers];
    // auto aterm             = new std::complex<float>[size_aterm];
    // auto aterm_offsets     = new int[nr_timeslots+1];
    auto spheroidal        = new float[size_spheroidal];
    auto spheroidal_double = new double[size_spheroidal];
    auto grid_double       = new std::complex<double>[size_grid];
    auto baselines         = new int[size_baselines];

    idg::init_example_uvw(uvw, nr_stations, nr_baselines, nr_time);
    idg::init_example_wavenumbers(wavenumbers, nr_channels);
    // idg::init_identity_aterm(aterm, nr_timeslots, nr_stations,
    //                          subgridsize, nr_polarizations);
    // idg::init_example_aterm_offsets(aterm_offsets, nr_timeslots, nr_time);
    idg::init_identity_spheroidal(spheroidal, subgridsize);
    idg::init_example_baselines(baselines, nr_stations, nr_baselines);

    double frequencyList[size_wavenumbers];
    for (auto chan = 0; chan < nr_channels; ++chan) {
        frequencyList[chan] = 299792458.0 * wavenumbers[chan] / (2 * M_PI);
    }
    for (auto k = 0; k < size_uvw; ++k) {
        uvw_double[k] = uvw[k];
    }

    for (auto k = 0; k < size_spheroidal; ++k) {
        spheroidal_double[k] = spheroidal[k];
    }

    grid_double[0*gridsize*gridsize + (gridsize/2)*gridsize + gridsize/2] = 1;
    grid_double[1*gridsize*gridsize + (gridsize/2)*gridsize + gridsize/2] = 1;
    grid_double[2*gridsize*gridsize + (gridsize/2)*gridsize + gridsize/2] = 1;
    grid_double[3*gridsize*gridsize + (gridsize/2)*gridsize + gridsize/2] = 1;

    std::clog << std::endl;

    auto bufferSize = nr_time;

    /////////////////////////////////////////////////////////////////////

    idg::DegridderPlan degridder(idg::Type::CPU_OPTIMIZED, bufferSize);
    degridder.set_stations(nr_stations);
    degridder.set_frequencies(nr_channels, frequencyList);
    degridder.set_grid(4, gridsize, gridsize, grid_double);
    // degridder.set_spheroidal(subgridsize, spheroidal_double);
    degridder.set_image_size(imagesize);
    degridder.set_w_kernel(subgridsize/2);
    degridder.internal_set_subgrid_size(subgridsize);
    degridder.bake();

    idg::GridderPlan gridder(idg::Type::CPU_OPTIMIZED, bufferSize);
    gridder.set_stations(nr_stations);
    gridder.set_frequencies(nr_channels, frequencyList);
    gridder.set_grid(4, gridsize, gridsize, grid_double);
    // gridder.set_spheroidal(subgridsize, spheroidal_double);
    gridder.set_image_size(imagesize);
    gridder.set_w_kernel(subgridsize/2);
    gridder.internal_set_subgrid_size(subgridsize);
    gridder.bake();

    /////////////////////////////////////////////////////////////////////

    degridder.transform_grid();

    /////////////////////////////////////////////////////////////////////

    for (auto time_batch = 0; time_batch < nr_time/bufferSize; ++time_batch) {

        for (auto time_minor = 0; time_minor < bufferSize; ++time_minor) {

            auto time = time_batch*bufferSize + time_minor;

            for (auto bl = 0; bl < nr_baselines; ++bl) {

                auto antenna1 = baselines[bl*2];
                auto antenna2 = baselines[bl*2 + 1];

                degridder.request_visibilities(
                    &uvw_double[bl*nr_time*3 + time*3],
                    antenna1,
                    antenna2,
                    time);
            }

        }

        degridder.flush();

        for (auto time_minor = 0; time_minor < bufferSize; ++time_minor) {

            auto time = time_batch*bufferSize + time_minor;

            for (auto bl = 0; bl < nr_baselines; ++bl) {

                auto antenna1 = baselines[bl*2];
                auto antenna2 = baselines[bl*2 + 1];

                degridder.read_visibilities(
                   antenna1,
                   antenna2,
                   time,
                   visibilities);

                gridder.grid_visibilities(
                    visibilities,
                    &uvw_double[bl*nr_time*3 + time*3],
                    antenna1,
                    antenna2,
                    time);

            }

        }

        gridder.flush();

    }

    /////////////////////////////////////////////////////////////////////

    delete [] visibilities;
    delete [] uvw;
    delete [] uvw_double;
    delete [] wavenumbers;
    // delete [] aterm;
    // delete [] aterm_offsets;
    delete [] spheroidal;
    delete [] spheroidal_double;
    delete [] grid_double;
    delete [] baselines;

    return 0;
}
