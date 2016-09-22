#include <stdexcept>
#include "MS-hdf5.h"

using namespace std;

namespace idg {

    typedef struct Complex64_t {
        float  r;
        float  i;
    } Complex64_t;

    MS_hdf5::MS_hdf5(std::string filename, unsigned int mode)
        : file(filename, mode)
    { }

    unsigned int MS_hdf5::get_nr_antennas()
    {
        unsigned int  nr_antennas;
        H5::Group     rootGroup         = file.openGroup( "/data" );
        H5::Attribute antennas_attr     = rootGroup.openAttribute("NR_ANTENNAS");
        antennas_attr.read(antennas_attr.getDataType(), &nr_antennas);
        return nr_antennas;
    }

    unsigned int MS_hdf5::get_nr_baselines()
    {
        unsigned int  nr_baselines;
        H5::Group     rootGroup         = file.openGroup( "/data" );
        H5::Attribute baselines_attr    = rootGroup.openAttribute("NR_BASELINES");
        baselines_attr.read(baselines_attr.getDataType(), &nr_baselines);
        return nr_baselines;
    }

    unsigned int MS_hdf5::get_nr_timesteps()
    {
        unsigned int  nr_timesteps;
        H5::Group     rootGroup         = file.openGroup( "/data" );
        H5::Attribute timesteps_attr    = rootGroup.openAttribute("NR_TIMESTEPS");
        timesteps_attr.read(timesteps_attr.getDataType(), &nr_timesteps);
        return nr_timesteps;
    }

    unsigned int MS_hdf5::get_nr_channels()
    {
        unsigned int  nr_channels;
        H5::Group     rootGroup         = file.openGroup( "/data" );
        H5::Attribute channel_attr      = rootGroup.openAttribute("NR_CHANNELS");
        channel_attr.read(channel_attr.getDataType(), &nr_channels);
        return nr_channels;
    }

    unsigned int MS_hdf5::get_nr_correlations()
    {
        unsigned int  nr_correlations;
        H5::Group     rootGroup         = file.openGroup( "/data" );
        H5::Attribute correlations_attr = rootGroup.openAttribute("NR_CORRELATIONS");
        correlations_attr.read(correlations_attr.getDataType(), &nr_correlations);
        return nr_correlations;
    }

    vector<double> MS_hdf5::read_frequencies()
    {
        vector<double> frequencies(get_nr_channels());

        H5::Group     rootGroup = file.openGroup( "/data" );
        H5::DataSet   dataset   = rootGroup.openDataSet( "FREQUENCIES" );
        H5::DataSpace dataspace = dataset.getSpace();

        dataset.read( frequencies.data(),
                      H5::PredType::NATIVE_DOUBLE );

        return frequencies;
    }

    // TODO: optinally read only for first timestep
    vector<int> MS_hdf5::read_antenna1()
    {
        vector<int> antenna1(get_nr_baselines() * get_nr_timesteps());

        H5::Group     rootGroup = file.openGroup( "/data" );
        H5::DataSet   dataset   = rootGroup.openDataSet( "ANTENNA1" );
        H5::DataSpace dataspace = dataset.getSpace();
        dataset.read( antenna1.data(), H5::PredType::NATIVE_INT );

        return antenna1;
    }

    // TODO: merge with get_antenna1()
    vector<int> MS_hdf5::read_antenna2()
    {
        vector<int> antenna2(get_nr_baselines() * get_nr_timesteps());

        H5::Group     rootGroup = file.openGroup( "/data" );
        H5::DataSet   dataset   = rootGroup.openDataSet( "ANTENNA2" );
        H5::DataSpace dataspace = dataset.getSpace();
        dataset.read( antenna2.data(), H5::PredType::NATIVE_INT );

        return antenna2;
    }


    idg::Grid3D<idg::Matrix2x2<complex<float>>>
    MS_hdf5::read_visibilities(int start_index, int timesteps)
    {
        idg::Grid3D<idg::Matrix2x2<complex<float>>> v(timesteps,
                                                      get_nr_baselines(),
                                                      get_nr_channels());

        // Define data type
        H5::CompType dtype( sizeof(Complex64_t) );
        dtype.insertMember( "r", HOFFSET(Complex64_t, r),
                            H5::PredType::NATIVE_FLOAT);
        dtype.insertMember( "i", HOFFSET(Complex64_t, i),
                            H5::PredType::NATIVE_FLOAT);

        // Select part of file to read
        H5::Group     rootGroup = file.openGroup( "/data" );
        H5::DataSet   dataset   = rootGroup.openDataSet( "DATA" );
        H5::DataSpace dataspace = dataset.getSpace();

        int rank = 4;
        hsize_t dims[rank];
        dataspace.getSimpleExtentDims(dims, NULL);

        hsize_t s = start_index;
        hsize_t start[4]  = {s, 0, 0, 0};
        hsize_t stride[4] = {1, 1, 1, 1};
        hsize_t count[4]  = {1, 1, 1, 1};
        hsize_t block[4];
        block[0] = timesteps;
        block[1] = dims[1];
        block[2] = dims[2];
        block[3] = dims[3];

        dataspace.selectHyperslab(H5S_SELECT_SET,
                                  count, start, stride, block);

        // Define a memory data space
        H5::DataSpace memspace(rank, block);

        // Read the data
        dataset.read( v.data(), dtype, memspace, dataspace );

        return v;
    }


    // TODO: merge with read_visibilities?
    idg::Grid3D<idg::Matrix2x2<int>>
    MS_hdf5::read_flags(int start_index, int timesteps)
    {
        idg::Grid3D<idg::Matrix2x2<int>> f(timesteps,
                                           get_nr_baselines(),
                                           get_nr_channels());

        // Select part of file to read
        H5::Group     rootGroup = file.openGroup( "/data" );
        H5::DataSet   dataset   = rootGroup.openDataSet( "FLAG" );
        H5::DataSpace dataspace = dataset.getSpace();

        int rank = 4;
        hsize_t dims[rank];
        dataspace.getSimpleExtentDims(dims, NULL);

        hsize_t s = start_index;
        hsize_t start[4]  = {s, 0, 0, 0};
        hsize_t stride[4] = {1, 1, 1, 1};
        hsize_t count[4]  = {1, 1, 1, 1};
        hsize_t block[4];
        block[0] = timesteps;
        block[1] = dims[1];
        block[2] = dims[2];
        block[3] = dims[3];

        dataspace.selectHyperslab(H5S_SELECT_SET,
                                  count, start, stride, block);

        // Define a memory data space
        H5::DataSpace memspace(rank, block);

        // Read the data
        dataset.read( f.data(), H5::PredType::NATIVE_INT, memspace, dataspace );

        return f;
    }


    idg::Grid2D<idg::UVWCoordinate<double>>
    MS_hdf5::read_uvw_coordinates(int start_index, int timesteps)
    {

        idg::Grid2D<idg::UVWCoordinate<double>> uvw(timesteps,
                                                    get_nr_baselines());

        // Select part of file to read
        H5::Group     rootGroup = file.openGroup( "/data" );
        H5::DataSet   dataset   = rootGroup.openDataSet( "UVW" );
        H5::DataSpace dataspace = dataset.getSpace();

        int rank = 3;
        hsize_t dims[rank];
        dataspace.getSimpleExtentDims(dims, NULL);

        hsize_t s = start_index;
        hsize_t start[3]  = {s, 0, 0};
        hsize_t stride[3] = {1, 1, 1};
        hsize_t count[3]  = {1, 1, 1};
        hsize_t block[3];
        block[0] = timesteps;
        block[1] = dims[1];
        block[2] = dims[2];

        dataspace.selectHyperslab(H5S_SELECT_SET,
                                  count, start, stride, block);

        // Define a memory data space
        H5::DataSpace memspace(rank, block);

        // Read the data
        dataset.read( uvw.data(), H5::PredType::NATIVE_DOUBLE, memspace, dataspace );

        return uvw;
    }


    void apply_flags(idg::Grid3D<idg::Matrix2x2<complex<float>>> & visibilities,
                     const idg::Grid3D<idg::Matrix2x2<int>>& flags)
    {
        size_t dim0 = visibilities.get_depth();
        size_t dim1 = visibilities.get_height();
        size_t dim2 = visibilities.get_width();

        if (dim0 != flags.get_depth() || dim1 != flags.get_height() || dim2 != flags.get_width())
            throw invalid_argument("Dimension mismatch.");

        for (auto i = 0; i < dim0; ++i) {
            for (auto j = 0; j < dim1; ++j) {
                for (auto k = 0; k < dim2; ++k) {
                    int* tmp = (int*) &flags(i,j,k);
                    if (tmp[0] == 1 || tmp[1] == 1 || tmp[2] == 1 || tmp[3] == 1) {
                        visibilities(i,j,k) = {0,0,0,0};
                    }
                }
            }
        }
    }

}
