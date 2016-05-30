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
    int nr_stations = params.get_nr_stations();
    int nr_baselines = params.get_nr_baselines();
    int nr_time =  params.get_nr_time();
    int nr_timeslots = params.get_nr_timeslots();
    int nr_channels = params.get_nr_channels();
    int gridsize = params.get_grid_size();
    int subgridsize = params.get_subgrid_size();
    float imagesize = params.get_imagesize();
    int nr_polarizations = params.get_nr_polarizations();

    float w_offset = 0;
    int kernel_size = (subgridsize / 4) + 1;

    // Print configuration
    std::clog << params;
    std::clog << std::endl;

    // Allocate and initialize data structures
    std::clog << ">>> Initialize data structures" << std::endl;

    auto size_visibilities = 1ULL * nr_baselines*nr_time*
        nr_channels*nr_polarizations;
    auto size_uvw = 1ULL * nr_baselines*nr_time*3;
    auto size_wavenumbers = 1ULL * nr_channels;
    auto size_aterm = 1ULL * nr_stations*nr_timeslots*
        nr_polarizations*subgridsize*subgridsize;
    auto size_spheroidal = 1ULL * subgridsize*subgridsize;
    auto size_grid = 1ULL * nr_polarizations*gridsize*gridsize;
    auto size_baselines = 1ULL * nr_baselines*2;

    auto visibilities = new std::complex<float>[size_visibilities];
    auto uvw = new float[size_uvw];
    auto uvw_double = new double[size_uvw];
    auto wavenumbers = new float[size_wavenumbers];
    auto aterm = new std::complex<float>[size_aterm];
    auto aterm_offsets = new int[nr_timeslots+1];
    auto spheroidal = new float[size_spheroidal];
    auto spheroidal_double = new double[size_spheroidal];
    auto grid_double = new std::complex<double>[size_grid];
    auto baselines = new int[size_baselines];

    idg::init_visibilities(visibilities, nr_baselines,
                           nr_time,
                           nr_channels, nr_polarizations);
    idg::init_uvw(uvw, nr_stations, nr_baselines, nr_time);
    idg::init_wavenumbers(wavenumbers, nr_channels);
    idg::init_aterm(aterm, nr_timeslots, nr_stations, subgridsize, nr_polarizations);
    idg::init_aterm_offsets(aterm_offsets, nr_timeslots, nr_time);
    idg::init_spheroidal(spheroidal, subgridsize);
    idg::init_baselines(baselines, nr_stations, nr_baselines);

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

    std::clog << std::endl;

    auto bufferSize = nr_time;

    /////////////////////////////////////////////////////////////////////

    // idg::GridderPlan plan(idg::Type::CPU_REFERENCE, bufferSize);
    // plan.set_stations(nr_stations);
    // plan.set_frequencies(frequencyList, nr_channels);
    // plan.set_grid(grid_double, 4, gridsize, gridsize);
    // plan.set_spheroidal(spheroidal_double, subgridsize);
    // plan.set_image_size(0.1);
    // plan.set_w_kernel(subgridsize/2);
    // plan.internal_set_subgrid_size(subgridsize);
    // plan.bake();


    // for (auto time = 0; time < nr_time; ++time) {
    //     for (auto bl = 0; bl < nr_baselines; ++bl) {

    //         auto antenna1 = baselines[bl*2];
    //         auto antenna2 = baselines[bl*2 + 1];

    //         // #if defined(DEBUG)
    //         // cout << "Adding: time " << time << ", "
    //         //      << "stations = (" << antenna1 << ", " << antenna2 << "), "
    //         //      << "uvw = ("
    //         //      << uvw_double[bl*nr_time*3 + time*3] << ", "
    //         //      << uvw_double[bl*nr_time*3 + time*3 + 1] << ", "
    //         //      << uvw_double[bl*nr_time*3 + time*3 + 2] << ")" << endl;
    //         // #endif

    //         plan.grid_visibilities(
    //             &visibilities[bl*nr_time*nr_channels*nr_polarizations +
    //                           time*nr_channels*nr_polarizations],
    //             &uvw_double[bl*nr_time*3 + time*3],
    //             antenna1,
    //             antenna2,
    //             time);
    //     }
    // }

    // plan.flush();

    /////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////

    idg::DegridderPlan plan(idg::Type::CPU_REFERENCE, bufferSize);
    plan.set_stations(nr_stations);
    plan.set_frequencies(frequencyList, nr_channels);
    plan.set_grid(grid_double, 4, gridsize, gridsize);
    plan.set_spheroidal(spheroidal_double, subgridsize);
    plan.set_image_size(0.1);
    plan.set_w_kernel(subgridsize/2);
    plan.internal_set_subgrid_size(subgridsize);
    plan.bake();

    std::vector<int> listOfRowIds;
    for (auto time = 0; time < nr_time; ++time) {
        for (auto bl = 0; bl < nr_baselines; ++bl) {

            auto antenna1 = baselines[bl*2];
            auto antenna2 = baselines[bl*2 + 1];

            // #if defined(DEBUG)
            // cout << "Adding: time " << time << ", "
            //      << "stations = (" << antenna1 << ", " << antenna2 << "), "
            //      << "uvw = ("
            //      << uvw_double[bl*nr_time*3 + time*3] << ", "
            //      << uvw_double[bl*nr_time*3 + time*3 + 1] << ", "
            //      << uvw_double[bl*nr_time*3 + time*3 + 2] << ")" << endl;
            // #endif

            plan.request_visibilities(
                &uvw_double[bl*nr_time*3 + time*3],
                antenna1,
                antenna2,
                time);
        }
    }

    plan.flush();

    /////////////////////////////////////////////////////////////////////



    return 0;
}
