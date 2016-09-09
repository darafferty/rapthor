#include <iostream>
#include <stdexcept>
#include <memory>
#include <vector>

#include "H5Cpp.h"
#include "idg.h"
#include "visualize.h"

using namespace std;


typedef struct complex64_t {
    int    r;
    float  i;
} complex64_t;


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
    H5::H5File file( ms_name, H5F_ACC_RDONLY );

    // Get attributes
    H5::Group     rootGroup         = file.openGroup( "/data" );
    H5::Attribute antennas_attr     = rootGroup.openAttribute("NR_ANTENNAS");
    H5::Attribute baselines_attr    = rootGroup.openAttribute("NR_BASELINES");
    H5::Attribute timesteps_attr    = rootGroup.openAttribute("NR_TIMESTEPS");
    H5::Attribute channel_attr      = rootGroup.openAttribute("NR_CHANNELS");
    H5::Attribute correlations_attr = rootGroup.openAttribute("NR_CORRELATIONS");

    size_t nr_antennas;
    size_t nr_baselines;
    size_t nr_timesteps;
    size_t nr_channels;
    size_t nr_correlations;

    antennas_attr.read(antennas_attr.getDataType(), &nr_antennas);
    baselines_attr.read(baselines_attr.getDataType(), &nr_baselines);
    timesteps_attr.read(timesteps_attr.getDataType(), &nr_timesteps);
    channel_attr.read(channel_attr.getDataType(), &nr_channels);
    correlations_attr.read(correlations_attr.getDataType(), &nr_correlations);

    #if defined(DEBUG)
    cout << endl;
    cout << "PARAMETERS READ FROM MS" << endl;
    cout << "-----------------------" << endl;
    cout << "Number of baselines:   " << nr_baselines    << endl;
    cout << "Number of correlation: " << nr_correlations << endl;
    cout << "Number of channels:    " << nr_channels     << endl;
    cout << "Number of antennas:    " << nr_antennas     << endl;
    cout << endl;
    #endif

    // Read frequencies
    H5::DataSet frequencies_dataset = rootGroup.openDataSet( "FREQUENCIES" );
    H5::DataSpace frequencies_dataspace = frequencies_dataset.getSpace();
    vector<double> frequencies(nr_channels);
    // frequencies_dataset.read( frequencies.data(),
    //                           H5::PredType::NATIVE_DOUBLE,
    //                           frequencies_dataspace,
    //                           frequencies_dataspace );
    frequencies_dataset.read( frequencies.data(),
                              H5::PredType::NATIVE_DOUBLE );

    // Read antenna IDs
    // TODO: read only the first "NR_BASELINES" values
    H5::DataSet antenna1_dataset = rootGroup.openDataSet( "ANTENNA1" );
    H5::DataSpace antenna1_dataspace = antenna1_dataset.getSpace();
    vector<int> antenna1(nr_baselines * nr_timesteps);
    // antenna1_dataset.read( antenna1.data(),
    //                        H5::PredType::NATIVE_INT,
    //                        antenna1_dataspace,
    //                        antenna1_dataspace );
    antenna1_dataset.read( antenna1.data(),
                           H5::PredType::NATIVE_INT );

    H5::DataSet antenna2_dataset = rootGroup.openDataSet( "ANTENNA2" );
    H5::DataSpace antenna2_dataspace = antenna2_dataset.getSpace();
    vector<int> antenna2(nr_baselines * nr_timesteps);
    // antenna2_dataset.read( antenna2.data(),
    //                        H5::PredType::NATIVE_INT,
    //                        antenna2_dataspace,
    //                        antenna2_dataspace );
    antenna2_dataset.read( antenna2.data(), H5::PredType::NATIVE_INT );


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
    cout << "-----------------------" << endl;
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
    gridder.set_frequencies(nr_channels, frequencies.data());
    gridder.set_grid(nr_polarizations, grid_size, grid_size, grid.get());
    gridder.set_cell_size(cell_size, cell_size);
    gridder.set_w_kernel(kernel_size);
    gridder.internal_set_subgrid_size(subgrid_size);
    gridder.bake();

    // Selection
    H5::DataSet   visibilities_dataset = rootGroup.openDataSet( "DATA" );
    H5::DataSpace visibilities_dataspace = visibilities_dataset.getSpace();
    int visibilities_rank = visibilities_dataspace.getSimpleExtentNdims();
    unique_ptr<hsize_t[]> visibilities_dims(new hsize_t[visibilities_rank]);
    visibilities_dataspace.getSimpleExtentDims(visibilities_dims.get(), NULL);

    H5::DataSet   flag_dataset = rootGroup.openDataSet( "FLAG" );
    H5::DataSpace flag_dataspace = flag_dataset.getSpace();
    int flag_rank = flag_dataspace.getSimpleExtentNdims();
    unique_ptr<hsize_t[]> flag_dims(new hsize_t[flag_rank]);
    flag_dataspace.getSimpleExtentDims(flag_dims.get(), NULL);

    H5::DataSet   uvw_dataset = rootGroup.openDataSet( "UVW" );
    H5::DataSpace uvw_dataspace = uvw_dataset.getSpace();
    int uvw_rank = uvw_dataspace.getSimpleExtentNdims();
    unique_ptr<hsize_t[]> uvw_dims(new hsize_t[uvw_rank]);
    uvw_dataspace.getSimpleExtentDims(uvw_dims.get(), NULL);

    #if defined(DEBUG)
    cout << "visibilities_rank = " << visibilities_rank << endl;
    cout << "visibilities_dims = ["
         << visibilities_dims[0] << ","
         << visibilities_dims[1] << ","
         << visibilities_dims[2] << ","
         << visibilities_dims[3] << "]" << endl;
    #endif

    H5::CompType mtype1( sizeof(complex64_t) );
    mtype1.insertMember( "r", HOFFSET(complex64_t, r),
                         H5::PredType::NATIVE_FLOAT);
    mtype1.insertMember( "i", HOFFSET(complex64_t, i),
                         H5::PredType::NATIVE_FLOAT);

    /* Select hyperslab for the dataset in the file */
    hsize_t visibilities_start[4]  = {0, 0, 0, 0};  // Start of hyperslab
    hsize_t visibilities_stride[4] = {1, 1, 1, 1}; // Stride of hyperslab
    hsize_t visibilities_count[4]  = {1, 1, 1, 1};  // Block count
    hsize_t visibilities_block[4];  // Block sizes
    visibilities_block[0] = 1;
    visibilities_block[1] = visibilities_dims[1];
    visibilities_block[2] = visibilities_dims[2];
    visibilities_block[3] = visibilities_dims[3];

    hsize_t uvw_start[3]  = {0, 0, 0};  // Start of hyperslab
    hsize_t uvw_stride[3] = {1, 1, 1}; // Stride of hyperslab
    hsize_t uvw_count[3]  = {1, 1, 1};  // Block count
    hsize_t uvw_block[3];  // Block sizes
    uvw_block[0] = 1;
    uvw_block[1] = uvw_dims[1];
    uvw_block[2] = uvw_dims[2];

    H5::DataSpace visibilities_memspace(
        3,
        &visibilities_block[1]);

    H5::DataSpace flag_memspace(
        3,
        &visibilities_block[1]);

    H5::DataSpace uvw_memspace(
        2,
        &uvw_block[1]);

    // TODO: read a block of data, instead of only one timestep?
    // TODO: use Grid3D etc data types?
    vector<double> uvw(nr_baselines * 3);
    vector<complex<float>>  visibilities(nr_baselines * nr_channels * nr_polarizations);
    vector<int> flags(nr_baselines * nr_channels * nr_polarizations);

    int number_timestips = min(nr_timesteps * (double)percentage/100, nr_timesteps);
    for (auto t = 0; t < number_timestips; ++t) {

        visibilities_start[0] = t;

        visibilities_dataspace.selectHyperslab(
            H5S_SELECT_SET,
            visibilities_count,
            visibilities_start,
            visibilities_stride,
            visibilities_block);

        visibilities_dataset.read( visibilities.data(),
                       mtype1,
                       visibilities_memspace,
                       visibilities_dataspace );

        flag_dataspace.selectHyperslab(
             H5S_SELECT_SET,
             visibilities_count,
             visibilities_start,
             visibilities_stride,
             visibilities_block);

        flag_dataset.read( flags.data(),
                     H5::PredType::NATIVE_INT,
                     flag_memspace,
                     flag_dataspace );

        // apply flags
        for (auto k = 0; k < nr_baselines * nr_channels * nr_polarizations; ++k) {
            if(flags[k] == 1) visibilities[k] = 0;
        }

        uvw_start[0] = t;

        uvw_dataspace.selectHyperslab(
            H5S_SELECT_SET,
            uvw_count,
            uvw_start,
            uvw_stride,
            uvw_block);

        uvw_dataset.read( uvw.data(),
                       H5::PredType::NATIVE_DOUBLE,
                       uvw_memspace,
                       uvw_dataspace );

        for (auto bl = 0; bl < nr_baselines; ++bl) {
        gridder.grid_visibilities(
            t,
            antenna1[bl],
            antenna2[bl],
            &uvw[3*bl],
            &visibilities[bl * nr_channels * nr_polarizations]);
        }

        // Display grid (note: equals one as buffer flushed iteration AFTER being full)
        if (t % buffer_timesteps == 1) {
             fig.display_matrix(grid_size, grid_size, grid.get(), "log", "jet");
        }
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
