#include "../common/common.h"

// Basic idea: write a bunch of test here on the reference code,
// and then make sure that all other implementation conform with
// the reference

using namespace std;

// Compare to analytical solution in case A-terms are identity and w=0
// This test covers the degridder without the A-term and w-terms computation
// See also test-degridder-001.py, which also visualizes the imaging
// of predicted visibilities
// TODO: this should be moved to Init.cpp and merged with "add_pt_src()"
void add_analytic_point_source(
    int            offset_x,
    int            offset_y,
    float          amplitude,
    float          image_size,
    int            nr_baselines,
    int            nr_time,
    int            nr_channels,
    int            nr_polarizations,
    int            grid_size,
    float          *uvw,
    float          *wavenumbers,
    complex<float> *true_visibilities)
{
    float l = offset_x * image_size / grid_size;
    float m = offset_y * image_size / grid_size;

    for (auto b=0; b<nr_baselines; ++b) {
        for (auto t=0; t<nr_time; ++t) {
            for (auto c=0; c<nr_channels; ++c) {
                float u = wavenumbers[c]*uvw[b*nr_time*3 + t*3 + 0]/(2*M_PI);
                float v = wavenumbers[c]*uvw[b*nr_time*3 + t*3 + 1]/(2*M_PI);
                complex<float> value = amplitude*exp(complex<float>(0,-2*M_PI*(u*l + v*m)));
                for (auto p=0; p<nr_polarizations; ++p) {
                    true_visibilities[b*nr_time*nr_channels*nr_polarizations +
                                      t*nr_channels*nr_polarizations +
                                      c*nr_polarizations + p] = value;
                 }
            }
        }
    }
}


int test01()
{
    int info = 0;

    // Set constants explicitly in the parameters parameter
    clog << ">>> Configuration"  << endl;
    idg::Parameters params;
    params.set_nr_stations(8);
    params.set_nr_channels(8);
    params.set_nr_time(4800);
    params.set_nr_timeslots(1);
    params.set_imagesize(0.08);
    params.set_grid_size(512);
    params.set_subgrid_size(24);

    // retrieve constants for memory allocation
    int nr_stations        = params.get_nr_stations();
    int nr_baselines       = params.get_nr_baselines();
    int nr_time            = params.get_nr_time();
    int nr_timeslots       = params.get_nr_timeslots();
    int nr_channels        = params.get_nr_channels();
    int gridsize           = params.get_grid_size();
    int subgridsize        = params.get_subgrid_size();
    float imagesize        = params.get_imagesize();
    int nr_polarizations   = params.get_nr_polarizations();
    float w_offset         = 0.0f;
    float integration_time = 2.0f;
    int kernel_size        = (subgridsize / 2) + 1;

    // error tolerance, which might need to be adjusted if parameters are changed
    float tol = 0.1f;

    // Print configuration
    clog << params;
    clog << endl;

    // Allocate and initialize data structures
    clog << ">>> Initialize data structures" << endl;

    auto size_visibilities = 1ULL * nr_baselines*nr_time*nr_channels*nr_polarizations;
    auto size_uvw          = 1ULL * nr_baselines*nr_time*3;
    auto size_wavenumbers  = 1ULL * nr_channels;
    auto size_aterm        = 1ULL * nr_timeslots*nr_stations*subgridsize*subgridsize*
                                    nr_polarizations;
    auto size_spheroidal   = 1ULL * subgridsize*subgridsize;
    auto size_grid         = 1ULL * nr_polarizations*gridsize*gridsize;
    auto size_baselines    = 1ULL * nr_baselines*2;

    auto visibilities      = new complex<float>[size_visibilities];
    auto true_visibilities = new complex<float>[size_visibilities];
    auto uvw               = new float[size_uvw];
    auto wavenumbers       = new float[size_wavenumbers];
    auto aterm             = new complex<float>[size_aterm];
    auto aterm_offsets     = new int[nr_timeslots+1];
    auto spheroidal        = new float[size_spheroidal];
    auto grid              = new complex<float>[size_grid];
    auto baselines         = new int[size_baselines];

    idg::init_example_uvw(uvw, nr_stations, nr_baselines, nr_time, integration_time);
    idg::init_example_wavenumbers(wavenumbers, nr_channels);
    idg::init_identity_aterm(aterm, nr_timeslots, nr_stations, subgridsize, nr_polarizations);
    idg::init_example_aterm_offsets(aterm_offsets, nr_timeslots, nr_time);
    idg::init_example_spheroidal(spheroidal, subgridsize);
    idg::init_zero_grid(grid, gridsize, nr_polarizations);
    idg::init_example_baselines(baselines, nr_stations, nr_baselines);

    // Set w-terms to zero
    for (auto i=0; i<size_uvw; i++) {
        if ((i+1)%3 == 0) uvw[i] = 0.0f;
    }

    // Initialize of center point source
    int   offset_x   = 80;
    int   offset_y   = 50;
    int   location_x = gridsize/2 + offset_x;
    int   location_y = gridsize/2 + offset_y;
    float amplitude  = 1.0f;
    grid[0*gridsize*gridsize + location_y*gridsize + location_x] = amplitude;
    grid[1*gridsize*gridsize + location_y*gridsize + location_x] = amplitude;
    grid[2*gridsize*gridsize + location_y*gridsize + location_x] = amplitude;
    grid[3*gridsize*gridsize + location_y*gridsize + location_x] = amplitude;

    add_analytic_point_source(
        offset_x, offset_y, amplitude,
        imagesize, nr_baselines, nr_time,
        nr_channels, nr_polarizations, gridsize,
        uvw, wavenumbers, true_visibilities);
    clog << endl;


    clog << ">>> Initialize proxy" << endl;

    idg::proxy::cpu::Reference proxy(params);


    clog << ">>> Predict visibilities" << endl;

    proxy.transform(idg::ImageDomainToFourierDomain, grid);

    proxy.degrid_visibilities(
        visibilities,
        uvw,
        wavenumbers,
        baselines,
        grid,
        w_offset,
        kernel_size,
        aterm,
        aterm_offsets,
        spheroidal);

    clog << endl;

    float error = get_accucary(size_visibilities,
                               visibilities,
                               true_visibilities);

    cout << "Error = " << error << endl;

    // Report results
    if (error < tol) {
        cout << "Prediction test PASSED!" << endl;
    } else {
        cout << "Prediction test FAILED!" << endl;
        info = 1;
    }

    // Free memory for data structures
    delete[] visibilities;
    delete[] true_visibilities;
    delete[] uvw;
    delete[] wavenumbers;
    delete[] aterm;
    delete[] aterm_offsets;
    delete[] spheroidal;
    delete[] grid;

    return info;
}



int main(int argc, char *argv[])
{
    int info = 0;

    info = test01();
    if (info != 0) return info;

    // info = test02();
    // if (info != 0) return info;

    // info = test03();
    // if (info != 0) return info;

    return info;
}
