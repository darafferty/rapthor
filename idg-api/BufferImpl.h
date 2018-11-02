/*
 * BufferImpl.h
 * Access to IDG's high level gridder routines
 */

#ifndef IDG_BUFFERIMPL_H_
#define IDG_BUFFERIMPL_H_

#include <complex>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include "idg-common.h"
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

#include "Buffer.h"


namespace idg {
namespace api {

    class BufferSetImpl;

    class BufferImpl : public virtual Buffer
    {
    public:

        // Constructors and destructor
        BufferImpl(
            BufferSetImpl* bufferset,
            proxy::Proxy* proxy,
            size_t bufferTimesteps = 4096);

        virtual ~BufferImpl();

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

        void set_w_step(float w_step);
        float get_w_step() const;
        
        void set_shift(const float* shift);
        const idg::Array1D<float>& get_shift() const;

        void set_kernel_size(float size);
        float get_kernel_size() const;

        void set_spheroidal(
            size_t size,
            const float* spheroidal);

        void set_spheroidal(
            size_t height,
            size_t width,
            const float* spheroidal);

        void set_grid(
            Grid* grid);

        void set_max_baseline(float max_baseline) {m_max_baseline = max_baseline;}

        void set_uv_span_frequency(float uv_span_frequency) {m_uv_span_frequency = uv_span_frequency;}

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

        virtual void finished() {}

        /** \brief Sets a new aterm for the buffer
         *  \param timeIndex [in] 0 <= timeIndex < NR_TIMESTEPS
         *                        or 0 <= timeIndex < bufferTimesteps
         *  \param aterm [in] std::complex<float>[nrStations][subgridsize][subgridsize]
         */
        virtual void set_aterm(
            size_t timeIndex,
            const std::complex<float>* aterms);

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

        BufferSetImpl *m_bufferset; // pointer to parent BufferSet

        // Bookkeeping
        size_t m_bufferTimesteps;
        size_t m_timeStartThisBatch;
        size_t m_timeStartNextBatch;
        std::set<size_t> m_timeindices;
        std::vector<std::pair<int, int>> m_channel_groups;

        //
        float m_max_baseline;
        float m_uv_span_frequency;

        // Parameters for proxy
        size_t m_nrStations;
        size_t m_nr_channels;
        size_t m_nr_baselines;
        size_t m_nrPolarizations;
        size_t m_gridHeight;
        size_t m_gridWidth;
        size_t m_nr_w_layers;
        float  m_cellHeight;
        float  m_cellWidth;
        float  m_wStepInLambda;
        Array1D<float> m_shift;
        float  m_kernel_size;
        std::vector<unsigned int>  m_default_aterm_offsets;
        std::vector<unsigned int>  m_aterm_offsets;
        Array1D<unsigned int> m_aterm_offsets_array;
        proxy::Proxy* m_proxy;

        // Buffers
        Array1D<float> m_frequencies;                               // CH
        std::vector<Array1D<float>> m_grouped_frequencies;          // CH
        Array2D<float> m_spheroidal;                                     // SB x SB
        std::vector<Matrix2x2<std::complex<float>>> m_aterms;
        std::vector<Matrix2x2<std::complex<float>>> m_default_aterms;
        Array4D<Matrix2x2<std::complex<float>>> m_aterms_array;     // ST x SB x SB

        Array2D<UVWCoordinate<float>> m_bufferUVW;                       // BL x TI
        Array1D<std::pair<unsigned int,unsigned int>> m_bufferStationPairs;                         // BL
        std::vector<Array3D<Visibility<std::complex<float>>>> m_bufferVisibilities;   // BL x TI x CH
        Array3D<Visibility<std::complex<float>>> m_visibilities;   // BL * TI * CH

        Grid* m_grid; // pointer grid

        // references to members of parent BufferSet
        size_t &m_subgridsize;

    };

} // namespace api
} // namespace idg

#endif
