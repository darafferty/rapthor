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

#include "idg-common.h"
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
        FourierToImage,
        ImageToFourier
    };

    class Scheme   // rename to Plan
    {
    public:
        // Constructors and destructor
        Scheme(Type architecture = Type::CPU_REFERENCE,
               size_t bufferTimesteps = 4096);

        virtual ~Scheme();

        // Set/get all parameters
        void set_frequencies(const double* frequencyList,
                             size_t channelCount);

        size_t get_frequencies_size() const;
        double get_frequency(size_t channel) const;

        void set_stations(size_t nrStations);
        size_t get_stations() const;

        // Deprecated: use cell size
        void set_image_size(double imageSize);
        double get_image_size() const;

        void set_w_kernel(size_t size);
        size_t get_w_kernel_size() const;

        void set_spheroidal(
            const double* spheroidal,
            size_t size);

        void set_spheroidal(
            const double* spheroidal,
            size_t height,
            size_t width);

        void start_w_layer(double layerWInLambda);
        void finish_w_layer();

        void set_grid(
            std::complex<double>* grid,
            size_t nr_polarizations,
            size_t size
            );

        void set_grid(
            std::complex<double>* grid,
            size_t nr_polarizations,
            size_t height,
            size_t width);

        // Bake the plan: initialize data structures etc.
        // Must be called before the plan is used if have settings have changed
        // after construction
        void bake();

        virtual void flush() = 0;

        void start_aterm(
            const std::complex<double>* aterm,
            size_t nrStations,
            size_t size,
            size_t nrPolarizations);

        void start_aterm(
            const std::complex<double>* aterm,
            size_t nrStations,
            size_t height,
            size_t width,
            size_t nrPolarizations);

        void finish_aterm();

        void transform_grid(Direction direction,
                            std::complex<double>* grid,
                            size_t nr_polarizations,
                            size_t height,
                            size_t width);

        // Internal function, not needed for most users
        void internal_set_subgrid_size(const size_t size);
        size_t internal_get_subgrid_size() const;

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
        float  m_wOffsetInMeters; // Q: meters? lambda?
        size_t m_nrPolarizations;
        size_t m_wKernelSize;
        size_t m_gridHeight;
        size_t m_gridWidth;
        size_t m_subgridSize;
        float  m_imageSize;
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
