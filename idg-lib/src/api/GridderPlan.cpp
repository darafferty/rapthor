/*
 * GridderPlan.h
 * Access to IDG's high level gridder routines
 */


#include "GridderPlan.h"


using namespace std;

namespace idg {

    // Constructors and destructor
    GridderPlan::GridderPlan(Type architecture,
                             size_t buffer_timesteps)
        : m_architecture(architecture),
          m_bufferTimesteps(buffer_timesteps),
          m_nrStations(2),
          m_nrGroups(1),
          m_startTimeIndex(0),
          m_wOffsetInMeters(0.0f),
          m_nrPolarizations(4),
          m_kernelSize(1),
          m_gridHeight(0),
          m_gridWidth(0),
          m_subgridSize(32),
          m_proxy(nullptr)
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif
    }

    GridderPlan::~GridderPlan()
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif
    }


    // Set/get all parameters

    void GridderPlan::set_stations(const size_t nrStations)
    {
        m_nrStations = nrStations;
        m_nrGroups = ((nrStations - 1) * nrStations) / 2;
    }


    size_t GridderPlan::get_stations() const
    {
        return m_nrStations;
    }


    void GridderPlan::set_subgrid_size(const size_t size)
    {
        m_subgridSize = size;
    }


    size_t GridderPlan::get_subgrid_size() const
    {
        return m_subgridSize;
    }


    void GridderPlan::set_frequencies(const double* frequencyList,
                                      size_t channelCount)
    {
        m_frequencies.reserve(channelCount);
        for (int i=0; i<channelCount; i++) {
            m_frequencies.push_back(frequencyList[i]);
        }

        const double SPEED_OF_LIGHT = 299792458.0;
        m_wavenumbers = m_frequencies;
        for (auto &x : m_wavenumbers) {
            x = 2 * M_PI * x / SPEED_OF_LIGHT;
        }

    }


    double GridderPlan::get_frequency(const size_t channel) const
    {
        return m_frequencies[channel];
    }


    size_t GridderPlan::get_frequencies_size() const
    {
        return m_frequencies.size();
    }


    void GridderPlan::start_w_layer(double layerWInLambda)
    {
        // TODO: !!!
        m_wOffsetInMeters = float(layerWInLambda); // in lambda or in meters?
    }


    void GridderPlan::finish_w_layer()
    {
        execute();
    }


    void GridderPlan::set_grid(std::complex<double>* grid,
                               size_t height, size_t width)
    {
        m_grid_double = grid;

        if (height != width)
            throw invalid_argument("Only square grids supported so far.");
        m_gridHeight = height;
        m_gridWidth  = width;
    }


    // Plan creation and helper functions

    void GridderPlan::bake()
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif

        // (1) Create new proxy
        delete m_proxy;

        Parameters params;
        params.set_nr_stations(m_nrStations);
        // int nr_baselines = params.get_nr_baselines();
        params.set_nr_time(m_bufferTimesteps);
        params.set_nr_timeslots(1); // TODO: Remove as compile time const
        params.set_nr_channels(m_frequencies.size()); // TODO: Remove as compile time const
        // params.set_grid_size(grid.size());
        params.set_grid_size(1024); // TODO: Remove as compile time const
        params.set_subgrid_size(m_subgridSize);
        params.set_imagesize(0.1); // HACK
        // params.set_nr_polarizations(4);

        // TODO: remove compile time parameter from proxy

        #if defined(BUILD_LIB_CPU)
        if (m_architecture == Type::CPU_REFERENCE)
            m_proxy = new proxy::cpu::Reference(params);
        else if (m_architecture == Type::CPU_OPTIMIZED)
            m_proxy = new proxy::cpu::HaswellEP(params);
        else
            throw invalid_argument("Unknown architecture type.");
        #endif

        // (2) Setup buffers
        m_bufferUVW.reserve(m_nrGroups, m_bufferTimesteps);
        m_bufferVisibilities.reserve(m_nrGroups, m_bufferTimesteps, get_frequencies_size());
        m_bufferStationPairs.reserve(m_nrGroups);
        m_grid.reserve(get_frequencies_size(), m_gridHeight, m_gridWidth);
    }


    /* The baseline index is formed such that:
     *   0 implies antenna1=0, antenna2=1 ;
     *   1 implies antenna1=0, antenna2=2 ;
     * n-1 implies antenna1=1, antenna2=2 etc. */
    size_t GridderPlan::baseline_index(size_t antenna1, size_t antenna2)
    {
        if (antenna1 > antenna2) {
            swap(antenna1, antenna2);
        }
        auto offset =  antenna1*m_nrStations - ((antenna1+1)*antenna1)/2 - 1;
        return antenna2 - antenna1 + offset;
    }


    // Gridding routines

    void GridderPlan::grid_visibilities(
        const complex<float>* visibilities, // size CH x PL
        const double* uvwInMeters,
        size_t antenna1,
        size_t antenna2,
        size_t timeIndex)
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif

        auto time = timeIndex - m_startTimeIndex;
        auto bl = baseline_index(antenna1, antenna2);

        if (time == m_bufferTimesteps) {
            execute();
            // use updated m_startTimeIndex
            time = timeIndex - m_startTimeIndex;
        }

        // Copy data into buffers
        m_bufferUVW(bl, time) = {uvwInMeters[0], uvwInMeters[1], uvwInMeters[2]};

        if (antenna1 > antenna2) swap(antenna1, antenna2);
        m_bufferStationPairs[bl] = {antenna1, antenna2};

        copy(visibilities, visibilities + get_frequencies_size() * m_nrPolarizations,
             (complex<float>*) &m_bufferVisibilities(bl, time, 0));
    }


    // Must be called whenever the buffer is full or no more data added
    void GridderPlan::execute()
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif

        // TODO: remove hack to ignore aterm
        // WARNING: layout does not match the one in the kernel
        m_aterms.reserve(m_nrStations, m_subgridSize, m_subgridSize); // HACK
        for (auto s = 0; s < m_nrStations; ++s)
            for (auto y = 0; y < m_subgridSize; ++y)
                for (auto x = 0; x < m_subgridSize; ++x)
                    m_aterms(s, y, x) = {complex<float>(1), complex<float>(0),
                                         complex<float>(0), complex<float>(1)};
        int aterm_offsets[2] = {0, m_bufferTimesteps};

        // TODO: remove hack to ignore spheroidal
        m_spheroidal.reserve(m_subgridSize, m_subgridSize);
        for (auto y = 0; y < m_subgridSize; ++y)
            for (auto x = 0; x < m_subgridSize; ++x)
                m_spheroidal(y, x) = 1.0;

        // TODO: this routine should be not much more than this call
        m_proxy->grid_visibilities((complex<float>*) m_bufferVisibilities.data(),
                                   (float*) m_bufferUVW.data(),
                                   (float*) m_wavenumbers.data(),
                                   (int*) m_bufferStationPairs.data(),
                                   (complex<float>*) m_grid.data(),
                                   m_wOffsetInMeters,
                                   m_kernelSize,
                                   (complex<float>*) m_aterms.data(),
                                   aterm_offsets,
                                   (float*) m_spheroidal.data());

        // HACK: Add results to double precision grid
        for (auto p=0; p<get_frequencies_size(); ++p)
            for (auto y=0; y<m_gridHeight; ++y)
                for (auto x=0; x<m_gridWidth; ++x)
                      m_grid_double[p*m_gridHeight*m_gridWidth
                                    + y*m_gridWidth + x] += m_grid(p, y, x);

        // Cleanup
        m_startTimeIndex = 0; // update here, not set to zero
    }

} // namespace idg
