#include "common.h"

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
    int            nr_timesteps,
    int            nr_channels,
    int            nr_correlations,
    int            grid_size,
    idg::Array2D<idg::UVWCoordinate<float>>& uvw,
    idg::Array1D<float>& frequencies,
    idg::Array3D<idg::Visibility<std::complex<float>>>& visibilities_ref)
{
    float l = offset_x * image_size / grid_size;
    float m = offset_y * image_size / grid_size;

    for (auto bl = 0; bl < nr_baselines; bl++) {
        for (auto t = 0; t <nr_timesteps; t++) {
            for (auto c = 0; c < nr_channels; c++) {
                const double speed_of_light = 299792458.0;
                float u = (frequencies(c) / speed_of_light) * uvw(bl, t).u;
                float v = (frequencies(c) / speed_of_light) * uvw(bl, t).v;
                complex<float> value = amplitude*exp(complex<float>(0,-2*M_PI*(u*l + v*m)));
                visibilities_ref(bl, t, c).xx = value;
                visibilities_ref(bl, t, c).xy = value;
                visibilities_ref(bl, t, c).yx = value;
                visibilities_ref(bl, t, c).yy = value;
            }
        }
    }
}


int test01()
{
    int info = 0;

    // Parameters
    unsigned int nr_correlations = 4;
    float w_offset               = 0;
    unsigned int nr_stations     = 8;
    unsigned int nr_channels     = 9;
    unsigned int nr_timesteps    = 2048;
    unsigned int nr_timeslots    = 1;
    float image_size             = 0.08;
    unsigned int grid_size       = 512;
    unsigned int subgrid_size    = 24;
    float cell_size              = image_size / grid_size;
    unsigned int kernel_size     = (subgrid_size / 2) + 1;
    unsigned int nr_baselines    = (nr_stations * (nr_stations - 1)) / 2;

    // Print parameters
    print_parameters(
        nr_stations, nr_channels, nr_timesteps, nr_timeslots,
        image_size, grid_size, subgrid_size, kernel_size);


    // error tolerance, which might need to be adjusted if parameters are changed
    float tol = 0.1f;

    // Allocate and initialize data structures
    clog << ">>> Initialize data structures" << endl;
    idg::Array1D<float> frequencies =
        idg::get_example_frequencies(nr_channels);
    idg::Array3D<idg::Visibility<std::complex<float>>> visibilities =
        idg::get_example_visibilities(nr_baselines, nr_timesteps, nr_channels);
    idg::Array3D<idg::Visibility<std::complex<float>>> visibilities_ref =
        idg::get_example_visibilities(nr_baselines, nr_timesteps, nr_channels);
    idg::Array1D<std::pair<unsigned int,unsigned int>> baselines =
        idg::get_example_baselines(nr_stations, nr_baselines);
    idg::Array2D<idg::UVWCoordinate<float>> uvw =
        idg::get_example_uvw(nr_stations, nr_baselines, nr_timesteps);
    idg::Array3D<std::complex<float>> grid =
        idg::get_zero_grid(nr_correlations, grid_size, grid_size);
    idg::Array3D<std::complex<float>> grid_ref =
        idg::get_zero_grid(nr_correlations, grid_size, grid_size);
    idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterms =
        idg::get_identity_aterms(nr_timeslots, nr_stations, subgrid_size, subgrid_size);
    idg::Array1D<unsigned int> aterms_offsets =
        idg::get_example_aterms_offsets(nr_timeslots, nr_timesteps);
    idg::Array2D<float> spheroidal =
        idg::get_identity_spheroidal(subgrid_size, subgrid_size);
    idg::Array1D<float> shift(3); // zero shift
    clog << endl;

    // Set w-terms to zero
    for (int bl = 0; bl < nr_baselines; bl++) {
        for (int t = 0; t < nr_timesteps; t++) {
            uvw(bl, t).w = 0.0f;
        }
    }

    // Initialize of center point source
    int   offset_x   = 80;
    int   offset_y   = 50;
    int   location_x = grid_size/2 + offset_x;
    int   location_y = grid_size/2 + offset_y;
    float amplitude  = 1.0f;
    grid(0, location_y, location_x) = amplitude;
    grid(1, location_y, location_x) = amplitude;
    grid(2, location_y, location_x) = amplitude;
    grid(3, location_y, location_x) = amplitude;

    add_analytic_point_source(
        offset_x, offset_y, amplitude,
        image_size, nr_baselines, nr_timesteps,
        nr_channels, nr_correlations, grid_size,
        uvw, frequencies, visibilities_ref);
    clog << endl;

    // Initialize proxy
    clog << ">>> Initialize proxy" << endl;
    idg::proxy::cpu::Reference proxy;

    // Predict visibilities
    clog << ">>> Predict visibilities" << endl;

    proxy.transform(idg::ImageDomainToFourierDomain, grid);

    proxy.degridding(
        w_offset, shift.data(), cell_size, kernel_size, subgrid_size,
        frequencies, visibilities, uvw, baselines,
        grid, aterms, aterms_offsets, spheroidal);

    clog << endl;

    float error = get_accucary(
        nr_baselines*nr_timesteps*nr_channels*nr_correlations,
        (std::complex<float> *) visibilities.data(),
        (std::complex<float> *) visibilities_ref.data());

    cout << "Error = " << error << endl;

    // Report results
    if (error < tol) {
        cout << "Prediction test PASSED!" << endl;
    } else {
        cout << "Prediction test FAILED!" << endl;
        info = 1;
    }

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
