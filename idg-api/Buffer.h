/*
 * Buffer.h
 * Access to IDG's high level gridder routines
 */

#ifndef IDG_BUFFER_H_
#define IDG_BUFFER_H_

#include <complex>
#include <vector>
#include <set>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <limits>
#include "idg-common.h"
#include "idg-fft.h"
#if defined(BUILD_LIB_CPU)
#include "idg-cpu.h"
#endif
#if defined(BUILD_LIB_CUDA)
#include "idg-cuda.h"
#endif
#if defined(BUILD_LIB_CUDA) && defined(BUILD_LIB_CPU)
#include "idg-hybrid-cuda.h"
#endif
#if defined(BUILD_LIB_OPENCL)
#include "idg-opencl.h"
#endif
#include "Datatypes.h"

namespace idg {
namespace api {    

    enum class Type {
        CPU_REFERENCE,
        CPU_OPTIMIZED,
        CUDA_GENERIC,
        OPENCL_GENERIC,
        HYBRID_CUDA_CPU_OPTIMIZED
    };


    enum class Direction {
        FourierToImage = -1,
        ImageToFourier = 1
    };


    class Buffer
    {
    public:

        // Constructors and destructor
        Buffer(Type architecture = Type::CPU_REFERENCE,
               size_t bufferTimesteps = 4096);

        virtual ~Buffer();

        // Set/get all parameters
        void set_frequencies(size_t channelCount,
                             const double* frequencyList);

        void set_frequencies(const std::vector<double> &frequency_list);

        size_t get_frequencies_size() const;
        double get_frequency(size_t channel) const;

        void set_stations(size_t nrStations);
        size_t get_stations() const;

        void set_cell_size(double height, double width);
        double get_cell_height() const;
        double get_cell_width() const;

        void set_kernel_size(size_t size);
        size_t get_kernel_size() const;

        void set_spheroidal(
            size_t size,
            const float* spheroidal);

        void set_spheroidal(
            size_t height,
            size_t width,
            const float* spheroidal);

        void set_grid(
            Grid* grid);

//         void set_grid(
//             size_t nr_polarizations,
//             size_t height,
//             size_t width,
//             std::complex<float>* grid);

        size_t get_grid_height() const;
        size_t get_grid_width() const;
        size_t get_nr_polarizations() const;

        void set_image(double *image) {}
        
        
        // Bake the plan after parameters are set
        // Must be called before the plan is used
        // if have settings have changed after construction
        void bake();

        // Flush the buffer explicitly
        virtual void flush() = 0;

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
        
        virtual bool request_visibilities(
            size_t rowId,
            size_t timeIndex,
            size_t antenna1,
            size_t antenna2,
            const double* uvwInMeters) {}

        bool request_visibilities(
            size_t timeIndex,
            size_t antenna1,
            size_t antenna2,
            const double* uvwInMeters) {}

        virtual std::vector<std::pair<size_t, std::complex<float>*>> compute() {}

        virtual void finished_reading() {}

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

        void set_subgrid_size(const size_t size);
        size_t get_subgrid_size() const;

        double get_image_size() const;

    protected:
        /* Helper function to map (antenna1, antenna2) -> baseline index
         * The baseline index is formed such that:
         *   0 implies antenna1=0, antenna2=1 ;
         *   1 implies antenna1=0, antenna2=2 ;
         * n-1 implies antenna1=1, antenna2=2 etc. */
        size_t baseline_index(size_t antenna1, size_t antenna2) const;

        // Other helper routines
        virtual void malloc_buffers();
        void reset_buffers();
        void set_uvw_to_infinity();
        void init_default_aterm();

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
        float  m_wStepInLambda;
        size_t m_kernel_size;
        Array1D<unsigned int>  m_aterm_offsets;
        proxy::Proxy* m_proxy;

        // Buffers
        Array1D<float> m_frequencies;                               // CH
        Array1D<float> m_wavenumbers;                               // CH
        Array2D<float> m_spheroidal;                                     // SB x SB
        Array4D<Matrix2x2<std::complex<float>>> m_aterms;                    // ST x SB x SB

        Array2D<UVWCoordinate<float>> m_bufferUVW;                       // BL x TI
        Array1D<std::pair<unsigned int,unsigned int>> m_bufferStationPairs;                         // BL
        Array3D<Visibility<std::complex<float>>> m_bufferVisibilities;   // BL x TI x CH

        Grid* m_grid; // pointer grid
    };

} // namespace api
} // namespace idg

#endif
