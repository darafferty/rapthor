#include "common.h"

// Basic idea: write a bunch of test here on the reference code,
// and then make sure that all other implementation conform with
// the reference

using namespace std;

// Compare to analytical solution in case A-terms are identity and w=0
// This test covers the degridder without the A-term and w-terms computation
// See also test-degridder-001.py, which also visualizes the imaging
// of predicted visibilities

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
    unsigned int grid_size       = 512;
    unsigned int subgrid_size    = 24;
    unsigned int kernel_size     = (subgrid_size / 2) + 1;
    unsigned int nr_baselines    = (nr_stations * (nr_stations - 1)) / 2;

    // Initialize Data object
    idg::Data data(grid_size);
    float image_size             = data.get_image_size();
    float cell_size              = image_size / grid_size;

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
    idg::Array2D<idg::UVWCoordinate<float>> uvw =
        idg::get_example_uvw(nr_stations, nr_baselines, nr_timesteps);
    idg::Array3D<idg::Visibility<std::complex<float>>> visibilities =
        idg::get_example_visibilities(uvw, frequencies, image_size, grid_size);
    idg::Array3D<idg::Visibility<std::complex<float>>> visibilities_ref =
        idg::get_example_visibilities(uvw, frequencies, image_size, grid_size);
    idg::Array1D<std::pair<unsigned int,unsigned int>> baselines =
        idg::get_example_baselines(nr_stations, nr_baselines);
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
    for (unsigned bl = 0; bl < nr_baselines; bl++) {
        for (unsigned t = 0; t < nr_timesteps; t++) {
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
    memset(visibilities_ref.data(), 0, visibilities_ref.bytes());
    add_pt_src(visibilities_ref, uvw, frequencies, image_size, grid_size, offset_x, offset_y, amplitude);
    clog << endl;

    // Initialize proxy
    clog << ">>> Initialize proxy" << endl;
    idg::proxy::cpu::Reference proxy;

    // Predict visibilities
    clog << ">>> Predict visibilities" << endl;

    proxy.transform(idg::ImageDomainToFourierDomain, grid);

    proxy.degridding(
        w_offset, shift, cell_size, kernel_size, subgrid_size,
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
