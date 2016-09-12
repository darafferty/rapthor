#ifndef IDG_MS_HDF5
#define IDG_MS_HDF5

#include <complex>
#include <vector>
#include "idg.h"
#include "H5Cpp.h"

namespace idg {

    class MS_hdf5 {

    public:
        // Modes: H5F_ACC_TRUNC, H5F_ACC_EXCL, H5F_ACC_RDONLY, H5F_ACC_RDWR
        MS_hdf5(std::string filename, unsigned int mode = H5F_ACC_RDONLY);

        // get/set attributes
        unsigned int get_nr_antennas();
        unsigned int get_nr_baselines();
        unsigned int get_nr_timesteps();
        unsigned int get_nr_channels();
        unsigned int get_nr_correlations();

        // get/set data
        std::vector<double> read_frequencies();
        std::vector<int>    read_antenna1();
        std::vector<int>    read_antenna2();

        // Read visibilites with time in [start_index, start_index+timesteps)
        idg::Grid3D<idg::Matrix2x2<std::complex<float>>>
        read_visibilities(int start_index, int timesteps=1);

        // Read flags with time in [start_index, start_index+timesteps)
        idg::Grid3D<idg::Matrix2x2<int>>
        read_flags(int start_index, int timesteps=1);

        // Read uvw coordinates with time in [start_index, start_index+timesteps)
        idg::Grid2D<idg::UVWCoordinate<double>>
        read_uvw_coordinates(int start_index, int timesteps=1);


    private:
        H5::H5File file;
    };


    // helper routine to multiply visibilities and flags
    // NOTE: for now only for specialized types
    void apply_flags(idg::Grid3D<idg::Matrix2x2<std::complex<float>>>& visibilities,
                     const idg::Grid3D<idg::Matrix2x2<int>>& flags);

}

#endif
