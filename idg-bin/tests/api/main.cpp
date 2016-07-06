#include <iostream>
#include <cmath>

#include "idg.h"
#include "idg-utility.h"  // Data init routines

using namespace std;


// computes max|A[i]-B[i]| / max|B[i]|
template <typename S, typename T>
float get_accucary(
    const int size,
    const S* A,
    const T* B)
{
    float max_abs_error = 0.0f;
    float max_ref_val = 0.0f;
    float max_val = 0.0f;
    for (int i=0; i<size; i++) {
        float abs_error = abs(T(A[i]) - B[i]);
        if ( abs_error > max_abs_error ) {
            max_abs_error = abs_error;
        }
        if (abs(B[i]) > max_ref_val) {
            max_ref_val = abs(B[i]);
        }
        if (abs(A[i]) > max_val) {
            max_val = abs(A[i]);
        }
    }
    if (max_ref_val == 0.0f) {
        if (max_val == 0.0f)
            // both grid are zero
            return 0.0f;
        else
            // refrence grid is zero, but computed grid not
            return std::numeric_limits<float>::infinity();
    } else {
        return max_abs_error / max_ref_val;
    }
}


int main(int argc, char *argv[])
{
    int info = 0;
    float tol = 1000*std::numeric_limits<float>::epsilon();

    // Set number of batch to divide time
    int nr_batches = 4;

    // Set constants explicitly in the parameters parameter
    std::clog << ">>> Configuration"  << std::endl;
    idg::Parameters params;
    params.set_nr_stations(8);
    params.set_nr_channels(8);
    params.set_nr_time(1024*nr_batches); // multiple of batch for now
    params.set_imagesize(0.08);
    params.set_grid_size(1024);

    // for proxy creation
    params.set_nr_timeslots(nr_batches);
    params.set_subgrid_size(32);
    // params.set_subgrid_size(8);

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
    clog << params;
    clog << endl;

    // Allocate and initialize data structures
    clog << ">>> Initialize data structures" << endl;

    auto size_visibilities = 1ULL * nr_channels*nr_polarizations;
    auto size_uvw          = 1ULL * nr_baselines*nr_time*3;
    auto size_wavenumbers  = 1ULL * nr_channels;
    auto size_aterm        = 1ULL * nr_timeslots*nr_stations*
                             subgridsize*subgridsize*nr_polarizations;
    auto size_spheroidal   = 1ULL * subgridsize*subgridsize;
    auto size_grid         = 1ULL * nr_polarizations*gridsize*gridsize;
    auto size_baselines    = 1ULL * nr_baselines*2;

    auto visibilities            = new complex<float>[size_visibilities];
    auto visibilities_reference  = new complex<float>[nr_baselines*nr_time*size_visibilities];
    auto uvw                     = new float[size_uvw];
    auto uvw_double              = new double[size_uvw];
    auto wavenumbers             = new float[size_wavenumbers];
    auto aterm                   = new complex<float>[size_aterm];
    auto aterm_offsets           = new int[nr_timeslots+1];
    auto spheroidal              = new float[size_spheroidal];
    auto spheroidal_double       = new double[size_spheroidal];
    auto baselines               = new int[size_baselines];
    auto image_reference         = new complex<float>[size_grid];
    auto grid_reference          = new complex<float>[size_grid];
    auto image_double            = new complex<double>[size_grid];
    auto grid_double             = new complex<double>[size_grid];

    idg::init_example_uvw(uvw, nr_stations, nr_baselines, nr_time);
    idg::init_example_wavenumbers(wavenumbers, nr_channels);
    idg::init_identity_aterm(aterm, nr_timeslots, nr_stations,
                             subgridsize, nr_polarizations);
    idg::init_example_aterm_offsets(aterm_offsets, nr_timeslots, nr_time);
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

    int offset_x = 0;
    int offset_y = 0;
    size_t index = (offset_y + gridsize/2)*gridsize + (offset_x + gridsize/2);
    image_double[0*gridsize*gridsize + index] = 1;
    image_double[1*gridsize*gridsize + index] = 1;
    image_double[2*gridsize*gridsize + index] = 1;
    image_double[3*gridsize*gridsize + index] = 1;
    for (auto k = 0; k < size_grid; ++k) {
        image_reference[k] = image_double[k];
    }

    std::clog << std::endl;

    auto bufferSize = nr_time / nr_batches;

    /////////////////////////////////////////////////////////////////////

    idg::DegridderPlan degridder(idg::Type::CPU_OPTIMIZED, bufferSize);
    degridder.set_stations(nr_stations);   // planSetStations()?
    degridder.set_frequencies(nr_channels, frequencyList);
    degridder.set_grid(4, gridsize, gridsize, image_double); // not needed in plan
    degridder.set_spheroidal(subgridsize, spheroidal_double); // not needed in plan
    degridder.set_image_size(imagesize);
    degridder.set_w_kernel(kernel_size);
    degridder.internal_set_subgrid_size(subgridsize);
    degridder.bake();  // bakePlan()

    idg::GridderPlan gridder(idg::Type::CPU_OPTIMIZED, bufferSize);
    gridder.set_stations(nr_stations);
    gridder.set_frequencies(nr_channels, frequencyList);
    gridder.set_grid(4, gridsize, gridsize, grid_double);
    gridder.set_spheroidal(subgridsize, spheroidal_double);
    gridder.set_image_size(imagesize);
    gridder.set_w_kernel(kernel_size);
    gridder.internal_set_subgrid_size(subgridsize);
    gridder.bake();

    /////////////////////////////////////////////////////////////////////

    degridder.transform_grid();

    for (auto time_batch = 0; time_batch < nr_time/bufferSize; ++time_batch) {

        for (auto time_minor = 0; time_minor < bufferSize; ++time_minor) {

            auto time = time_batch*bufferSize + time_minor;

            for (auto bl = 0; bl < nr_baselines; ++bl) {

                auto antenna1 = baselines[bl*2];
                auto antenna2 = baselines[bl*2 + 1];

                degridder.request_visibilities(
                    time_minor,  // or 'time'
                    antenna1,
                    antenna2,
                    &uvw_double[bl*nr_time*3 + time*3]);
            }

        }

        degridder.flush();

        for (auto time_minor = 0; time_minor < bufferSize; ++time_minor) {

            auto time = time_batch*bufferSize + time_minor;

            for (auto bl = 0; bl < nr_baselines; ++bl) {

                auto antenna1 = baselines[bl*2];
                auto antenna2 = baselines[bl*2 + 1];

                degridder.read_visibilities(
                   time_minor,   // or 'time'
                   antenna1,
                   antenna2,
                   visibilities);

                gridder.grid_visibilities(
                    time_minor, // or 'time'
                    antenna1,
                    antenna2,
                    &uvw_double[bl*nr_time*3 + time*3],
                    visibilities);
            }

        }

        gridder.flush();

    }

    gridder.transform_grid();

    /////////////////////////////////////////////////////////////////////

    idg::proxy::cpu::HaswellEP proxy(params);

    clog << ">>> Run reference predict" << endl;
    proxy.transform(idg::ImageDomainToFourierDomain, image_reference);
    proxy.degrid_visibilities(visibilities_reference, uvw, wavenumbers, baselines,
                              image_reference, w_offset, kernel_size,
                              aterm, aterm_offsets, spheroidal);

    clog << ">>> Run reference imaging" << endl;
    proxy.grid_visibilities(visibilities_reference, uvw, wavenumbers, baselines,
                            grid_reference, w_offset, kernel_size,
                            aterm, aterm_offsets, spheroidal);
    proxy.transform(idg::FourierDomainToImageDomain, grid_reference);

    /////////////////////////////////////////////////////////////////////

    float error = get_accucary(size_grid, grid_double, grid_reference);

    // Report results
    if (error < tol) {
        cout << "WSClean API test PASSED! Error = " << error << endl;
    } else {
        cout << "WSClean API test FAILED! Error = " << error << endl;
        info = 1;
    }

    /////////////////////////////////////////////////////////////////////

    delete [] visibilities;
    delete [] visibilities_reference;
    delete [] uvw;
    delete [] uvw_double;
    delete [] wavenumbers;
    delete [] aterm;
    delete [] aterm_offsets;
    delete [] spheroidal;
    delete [] spheroidal_double;
    delete [] image_reference;
    delete [] grid_reference;
    delete [] image_double;
    delete [] grid_double;
    delete [] baselines;

    return info;
}
