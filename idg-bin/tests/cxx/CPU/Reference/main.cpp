#include "idg-cpu.h" // Reference proxy

using ProxyType = idg::proxy::cpu::Reference;

#include "common.h"

// Basic idea: write a bunch of test here on the reference code,
// and then make sure that all other implementation conform with
// the reference

using namespace std;

// Compare to analytical solution in case A-terms are identity and w=0
// This test covers the degridder without the A-term and w-terms computation

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
    unsigned int grid_size       = 1024;
    unsigned int subgrid_size    = 32;
    unsigned int kernel_size     = 9;
    unsigned int nr_baselines    = (nr_stations * (nr_stations - 1)) / 2;

    // Initialize Data object
    idg::Data data;

    // Determine the max baseline length for given grid_size
    auto max_uv = data.compute_max_uv(grid_size);

    // Select only baselines up to max_uv meters long
    data.limit_max_baseline_length(max_uv);
    data.print_info();

    // Restrict the number of baselines to nr_baselines
    data.limit_nr_baselines(nr_baselines);
    data.print_info();

    // Get remaining parameters
    auto image_size             = data.compute_image_size(grid_size);
    double cell_size            = image_size / grid_size;

    // Print parameters
    print_parameters(
        nr_stations, nr_channels, nr_timesteps, nr_timeslots,
        image_size, grid_size, subgrid_size, kernel_size);

    // error tolerance, which might need to be adjusted if parameters are changed
    float tol = 0.1f;

    // Allocate and initialize data structures
    clog << ">>> Initialize data structures" << endl;
    idg::Array1D<float> frequencies(nr_channels);
        data.get_frequencies(frequencies, image_size);
    idg::Array2D<idg::UVW<float>> uvw =
        data.get_uvw(nr_baselines, nr_timesteps);
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
    grid_ref(0, location_y, location_x) = amplitude;
    grid_ref(1, location_y, location_x) = amplitude;
    grid_ref(2, location_y, location_x) = amplitude;
    grid_ref(3, location_y, location_x) = amplitude;
    visibilities_ref.zero();
    add_pt_src(visibilities_ref, uvw, frequencies, image_size, grid_size, offset_x, offset_y, amplitude);
    clog << endl;

    // Initialize proxy
    clog << ">>> Initialize proxy" << endl;
    idg::proxy::cpu::Reference proxy;

    // Set grid
    idg::Grid grid_(grid.data(), 1, nr_correlations, grid_size, grid_size);
    proxy.set_grid(grid_);

    // Create plan
    clog << ">>> Create plan" << endl;
    idg::Plan::Options options;
    options.plan_strict = true;
    idg::Plan plan(
        kernel_size, subgrid_size, grid_size, cell_size,
        frequencies, uvw, baselines, aterms_offsets, options);
    clog << endl;

    // Grid reference visibilities
    clog << ">>> Grid visibilities" << endl;
    proxy.gridding(
        plan, w_offset, shift, cell_size, kernel_size, subgrid_size,
        frequencies, visibilities_ref, uvw, baselines,
        grid_, aterms, aterms_offsets, spheroidal);
    proxy.transform(idg::FourierDomainToImageDomain, grid);

    float grid_error = get_accuracy(
        grid_size*grid_size*nr_correlations,
        (std::complex<float> *) grid.data(),
        (std::complex<float> *) grid_ref.data());

    // Predict visibilities
    clog << ">>> Predict visibilities" << endl;

    proxy.transform(idg::ImageDomainToFourierDomain, grid_ref);

    // Set reference grid
    idg::Grid grid_ref_(grid_ref.data(), 1, nr_correlations, grid_size, grid_size);
    proxy.set_grid(grid_ref_);

    proxy.degridding(
        plan, w_offset, shift, cell_size, kernel_size, subgrid_size,
        frequencies, visibilities, uvw, baselines,
        grid_ref_, aterms, aterms_offsets, spheroidal);
    clog << endl;

    // Compute error
    float degrid_error = get_accuracy(
        nr_baselines*nr_timesteps*nr_channels*nr_correlations,
        (std::complex<float> *) visibilities.data(),
        (std::complex<float> *) visibilities_ref.data());

    // Report error
    clog << "Grid error = " << std::scientific << grid_error << endl;
    clog << "Degrid error = " << std::scientific << degrid_error << endl;
    clog << endl;


    // Report gridding results
    if (grid_error < tol) {
        cout << "Gridding test PASSED!" << endl;
    } else {
        cout << "Gridding test FAILED!" << endl;
        info = 1;
    }

    // Report degridding results
    if (degrid_error < tol) {
        cout << "Degridding test PASSED!" << endl;
    } else {
        cout << "Degridding test FAILED!" << endl;
        info = 2;
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
