/*
 * GridderPlan.h
 * Access to IDG's high level gridder routines
 */

#ifndef IDG_GRIDDERPLAN_H_
#define IDG_GRIDDERPLAN_H_

#include <complex>
#include <vector>
#include <stdexcept>
#include <cmath>

#include "idg-common.h"
#if defined(BUILD_LIB_CPU)
#include "idg-cpu.h"
#endif
#include "API.h"
#include "Datatypes.h"

namespace idg {

    class GridderPlan : public WSCLEAN_API
    {
    public:
        // Constructors and destructor
        GridderPlan(Type architecture = Type::CPU_REFERENCE,
                    size_t bufferTimesteps = 4096);

        // Q: Set everything in constructor and bake()?
        // GridderPlan(Type architecture = Type::CPU_REFERENCE,
        //             size_t bufferTimesteps = 4096, ...);

        virtual ~GridderPlan();

        // Set/get all parameters
        virtual void set_frequencies(const double* frequencyList,
                                     size_t channelCount) override;

        virtual size_t get_frequencies_size() const;
        virtual double get_frequency(const size_t channel) const;

        virtual void set_stations(const size_t nrStations) override;
        virtual size_t get_stations() const;

        virtual void set_image_size(const double imageSize) override;
        virtual double get_image_size() const;

        virtual void set_subgrid_size(const size_t size);
        virtual size_t get_subgrid_size() const;

        virtual void set_w_kernel(size_t size) override;
        virtual size_t get_w_kernel_size() const;

        virtual void set_spheroidal(
            const double* spheroidal,
            const size_t size);

        virtual void set_spheroidal(
            const double* spheroidal,
            const size_t height,
            const size_t width) override;


        virtual void start_w_layer(double layerWInLambda) override;

        virtual void finish_w_layer() override;

        virtual void set_grid(
            std::complex<double>* grid,
            const size_t nr_polarizations,
            const size_t size
            );

        virtual void set_grid(
            std::complex<double>* grid,
            const size_t nr_polarizations,
            const size_t height,
            const size_t width
            ) override;

        // Bake the plan: initialize data structures etc.
        // Must be called before the plan is used if have settings have changed
        // after construction
        virtual void bake() override;

        // Adds the visibilities to the buffer and then eventually to the grid
        virtual void grid_visibilities(
            const std::complex<float>* visibilities, // size CH x PL
            const double* uvwInMeters,               // (u, v, w)
            size_t antenna1,                         // 0 <= antenna1 < nrStations
            size_t antenna2,                         // antenna1 < antenna2 < nrStations
            size_t timeIndex) override;              // 0 <= timeIndex < NR_TIMESTEPS

        virtual void start_aterm(
            const std::complex<double>* aterm,
            const size_t nrStations,
            const size_t size);

        virtual void start_aterm(
            const std::complex<double>* aterm,
            const size_t nrStations,
            const size_t height,
            const size_t width) override;

        virtual void finish_aterm() override;

        // Must be called to flush the buffer
        virtual void execute() override;

    protected:
        /* Helper function to map (antenna1, antenna2) -> baseline index
         * The baseline index is formed such that:
         *   0 implies antenna1=0, antenna2=1 ;
         *   1 implies antenna1=0, antenna2=2 ;
         * n-1 implies antenna1=1, antenna2=2 etc. */
        size_t baseline_index(size_t antenna1, size_t antenna2);

        /* Helper routine to allocate buffers for the gridding stage */
        void malloc_gridder_buffers();
        void free_gridder_buffers();



    private:
        Type   m_architecture;
        size_t m_bufferTimesteps;
        size_t m_nrStations;
        size_t m_nrGroups;
        size_t m_startTimeIndex;
        float  m_wOffsetInMeters; // Q: meters? lambda?
        size_t m_nrPolarizations;
        size_t m_wKernelSize;
        size_t m_gridHeight;
        size_t m_gridWidth;
        size_t m_subgridSize;
        float  m_imageSize;
        int    m_aterm_offsets[2];

        std::vector<float> m_frequencies;                               // CH
        std::vector<float> m_wavenumbers;                               // CH
        Grid2D<UVWCoordinate<float>> m_bufferUVW;                       // BL x TI
        std::vector<StationPair<int>> m_bufferStationPairs;             // BL
        Grid3D<Visibility<std::complex<float>>> m_bufferVisibilities;   // BL x TI x CH
        Grid3D<Aterm<std::complex<float>>> m_aterms; // HACK: ST x SB x SB
        Grid2D<float> m_spheroidal;
        Grid3D<std::complex<float>> m_grid;          // HACK: complex<float> as needed for kernel
        std::complex<double>* m_grid_double;         // HACK: pointer to double precision grid
        proxy::Proxy* m_proxy;
    };

} // namespace idg

#endif
