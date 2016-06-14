/*
 * GridderPlan.h
 * Access to IDG's high level gridder routines
 */

#ifndef IDG_SCHEME_H_
#define IDG_SCHEME_H_

#include <complex>
#include <vector>
#include <set>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <fftw3.h>    // TODO: remove
#include "idg-common.h"
#include "idg-fft.h"
#if defined(BUILD_LIB_CPU)
#include "idg-cpu.h"
#endif
#include "API.h"
#include "Datatypes.h"

namespace idg {

    enum class Type {
        CPU_REFERENCE,
        CPU_OPTIMIZED,
        CUDA_KEPLER,
        CUDA_MAXWELL
    };


    enum class Direction {
        FourierToImage = -1,
        ImageToFourier = 1
    };


    class Scheme   // TODO: rename to Plan or so
    {
    public:

        // Constructors and destructor
        Scheme(Type architecture = Type::CPU_REFERENCE,
               size_t bufferTimesteps = 4096);

        virtual ~Scheme();

        // Set/get all parameters
        void set_frequencies(size_t channelCount,
                             const double* frequencyList);

        size_t get_frequencies_size() const;
        double get_frequency(size_t channel) const;

        void set_stations(size_t nrStations);
        size_t get_stations() const;

        void set_cell_size(double height, double width);
        double get_cell_height() const;
        double get_cell_width() const;

        void set_w_kernel(size_t size);
        size_t get_w_kernel_size() const;

        void set_spheroidal(
            size_t size,
            const double* spheroidal);

        void set_spheroidal(
            size_t height,
            size_t width,
            const double* spheroidal);

        void set_grid(
            size_t nr_polarizations,
            size_t size,
            std::complex<double>* grid);

        void set_grid(
            size_t nr_polarizations,
            size_t height,
            size_t width,
            std::complex<double>* grid);

        size_t get_grid_height() const;
        size_t get_grid_width() const;
        size_t get_nr_polarizations() const;

        // Bake the plan after parameters are set
        // Must be called before the plan is used
        // if have settings have changed after construction
        void bake();

        // Flush the buffer explicitly
        virtual void flush() = 0;


        // While filling buffer, start and end regions having
        // the same A-terms and same w-offset
        // Calling "start/finish" will flush the non-empty buffer implicitly
        void start_w_layer(double wOffsetInLambda);

        void finish_w_layer();

        void start_aterm(
            size_t nrStations,
            size_t size,
            size_t nrPolarizations,
            const std::complex<double>* aterm);

        void start_aterm(
            size_t nrStations,
            size_t height,
            size_t width,
            size_t nrPolarizations,
            const std::complex<double>* aterm);

        void finish_aterm();

        // Methods the transform the grid. i.e., perform FFT, scaling,
        // and apply the sheroidal
        virtual void transform_grid(
            double crop_tol            = 5e-3,
            size_t nr_polarizations    = 0,
            size_t height              = 0,
            size_t width               = 0,
            std::complex<double> *grid = nullptr) = 0;

        void fft_grid(
            size_t nr_polarizations    = 0,
            size_t height              = 0,
            size_t width               = 0,
            std::complex<double> *grid = nullptr);

        void ifft_grid(
            size_t nr_polarizations    = 0,
            size_t height              = 0,
            size_t width               = 0,
            std::complex<double> *grid = nullptr);

        void copy_grid(
            size_t nr_polarizations,
            size_t height,
            size_t width,
            std::complex<double>* grid);

        // Internal or deprecated methods

        // Internal function, not needed for most users
        void internal_set_subgrid_size(const size_t size);
        size_t internal_get_subgrid_size() const;

        // Deprecated: use cell size
        void set_image_size(double imageSize);
        double get_image_size() const;

    protected:
        /* Helper function to map (antenna1, antenna2) -> baseline index
         * The baseline index is formed such that:
         *   0 implies antenna1=0, antenna2=1 ;
         *   1 implies antenna1=0, antenna2=2 ;
         * n-1 implies antenna1=1, antenna2=2 etc. */
        size_t baseline_index(size_t antenna1, size_t antenna2) const;

        // Other helper routines
        void malloc_buffers();
        void reset_buffers();
        void set_uvw_to_infinity();
        void init_default_aterm();

        void fft_kernel(Direction direction,
                        size_t nr_polarizations = 0,
                        size_t height = 0,
                        size_t width = 0,
                        std::complex<double> *grid = nullptr);

        // Bookkeeping
        Type   m_architecture;
        size_t m_bufferTimesteps;
        size_t m_timeStartThisBatch;
        size_t m_timeStartNextBatch;
        std::set<size_t> m_timeindices;

        // Parameters for proxy
        size_t m_nrStations;
        size_t m_nrGroups;
        size_t m_nrPolarizations;
        size_t m_gridHeight;
        size_t m_gridWidth;
        size_t m_subgridSize;
        float  m_cellHeight;
        float  m_cellWidth;
        float  m_wOffsetInLambda;
        size_t m_wKernelSize;
        float  m_imageSize; // TODO: deprecated, remove member
        int    m_aterm_offsets[2];
        proxy::Proxy* m_proxy;

        // Buffers
        Frequencies<float> m_frequencies;                               // CH
        Wavenumbers<float> m_wavenumbers;                               // CH
        Grid2D<float> m_spheroidal;                                     // SB x SB
        Grid3D<Aterm<std::complex<float>>> m_aterms;                    // ST x SB x SB

        Grid2D<UVWCoordinate<float>> m_bufferUVW;                       // BL x TI
        StationPairs<int> m_bufferStationPairs;                         // BL
        Grid3D<Visibility<std::complex<float>>> m_bufferVisibilities;   // BL x TI x CH

        std::complex<double>* m_grid_double; // HACK: pointer to double precision grid
        Grid3D<std::complex<float>> m_grid;  // HACK: complex<float> as needed for kernel
    };

} // namespace idg

#endif
