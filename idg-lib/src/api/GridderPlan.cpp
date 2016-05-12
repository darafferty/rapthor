/*
 * GridderPlan.h
 * Access to IDG's high level gridder routines
 */


#include "GridderPlan.h"


using namespace std;

namespace idg {

    // Constructors and destructor
    GridderPlan::GridderPlan(Type architecture,
                             size_t bufferTimesteps)
        : m_architecture(architecture),
          m_bufferTimesteps(bufferTimesteps),
          m_nrStations(2),
          m_nrGroups(1),
          m_startTimeIndex(0),
          m_wOffsetInMeters(0.0f),
          m_nrPolarizations(4),
          m_wKernelSize(1),
          m_gridHeight(0),
          m_gridWidth(0),
          m_subgridSize(32),
          m_proxy(nullptr)
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif

        m_aterm_offsets[0] = 0;
        m_aterm_offsets[1] = bufferTimesteps;
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


    void GridderPlan::set_image_size(const double imageSize)
    {
        m_imageSize = float(imageSize);
    }


    double GridderPlan::get_image_size() const
    {
        return m_imageSize;
    }


    void GridderPlan::set_subgrid_size(const size_t size)
    {
        m_subgridSize = size;
    }


    size_t GridderPlan::get_subgrid_size() const
    {
        return m_subgridSize;
    }


    void GridderPlan::set_w_kernel(size_t size)
    {
        m_wKernelSize = size;
    }


    size_t GridderPlan::get_w_kernel() const
    {
        return m_wKernelSize;
    }


    void GridderPlan::set_spheroidal(
        const double* spheroidal,
        const size_t size)
    {
        set_spheroidal(spheroidal, size, size);
    }


    void GridderPlan::set_spheroidal(
        const double* spheroidal,
        const size_t height,
        const size_t width)
    {
        m_spheroidal.reserve(height, width);
        for (auto y = 0; y < height; ++y) {
            for (auto x = 0; x < width; x++) {
                m_spheroidal(y, x) = float(spheroidal[y*width + x]);
            }
        }
    }


    void GridderPlan::set_frequencies(
        const double* frequencyList,
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
        m_wOffsetInMeters = float(layerWInLambda); // Q: in lambda or in meters?
    }


    void GridderPlan::finish_w_layer()
    {
        execute();
    }


    void GridderPlan::set_grid(
        std::complex<double>* grid,
        const size_t size)
    {
        set_grid(grid, size, size);
    }


    void GridderPlan::set_grid(
        std::complex<double>* grid,
        const size_t height,
        const size_t width)
    {
        m_grid_double = grid;

        if (height != width)
            throw invalid_argument("Only square grids supported.");

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
        params.set_nr_time(m_bufferTimesteps);
        params.set_nr_timeslots(1);
        params.set_nr_channels( get_frequencies_size() ); // TODO: remove as compile time const
        params.set_grid_size(m_gridHeight);               // TODO: support non-square
        params.set_subgrid_size(m_subgridSize);
        params.set_imagesize(m_imageSize);                // TODO: remove as compile time const

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
        m_aterms.reserve(m_nrStations, m_subgridSize, m_subgridSize);
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


    void GridderPlan::start_aterm(
        const std::complex<double>* aterm,
        const size_t nrStations,
        const size_t height,
        const size_t width)
    {
        if (nrStations != m_nrStations)
            throw invalid_argument("Only square grids supported.");

        // to be implemented
        // TODO: remove hack to ignore aterm
        // WARNING: layout does not match the one in the kernel
        // TODO: Resize to SUBGRIDSIZE x SUBGRIDSIZE on the fly
        for (auto s = 0; s < m_nrStations; ++s)
            for (auto y = 0; y < m_subgridSize; ++y)
                for (auto x = 0; x < m_subgridSize; ++x)
                    m_aterms(s, y, x) = {complex<float>(1), complex<float>(0),
                                         complex<float>(0), complex<float>(1)};
    }


    void GridderPlan::start_aterm(
        const std::complex<double>* aterm,
        const size_t nrStations,
        const size_t size)
    {
        start_aterm(aterm, nrStations, size, size);
    }


    void GridderPlan::finish_aterm()
    {
        // to be implemented
        execute();
    }



    // Must be called whenever the buffer is full or no more data added
    void GridderPlan::execute()
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif

        int kernelsize = m_wKernelSize;

        // TODO: this routine should be not much more than this call
        m_proxy->grid_visibilities(
            (complex<float>*) m_bufferVisibilities.data(),
            (float*) m_bufferUVW.data(),
            (float*) m_wavenumbers.data(),
            (int*) m_bufferStationPairs.data(),
            (complex<float>*) m_grid.data(),
            m_wOffsetInMeters,
            kernelsize,
            (complex<float>*) m_aterms.data(),
            m_aterm_offsets,
            (float*) m_spheroidal.data());

        // HACK: Add results to double precision grid
        for (auto p=0; p<get_frequencies_size(); ++p) {
            for (auto y=0; y<m_gridHeight; ++y) {
                for (auto x=0; x<m_gridWidth; ++x) {
                    m_grid_double[p*m_gridHeight*m_gridWidth
                                  + y*m_gridWidth + x] += m_grid(p, y, x);
                }
            }
        }

        // Cleanup
        m_startTimeIndex = 0; // update here, not set to zero
    }

} // namespace idg



// C interface:
// Rationale: calling the code from C code and Fortran easier,
// and bases to create interface to scripting languages such as
// Python, Julia, Matlab, ...
extern "C" {

    idg::GridderPlan* GridderPlan_init(unsigned int bufferTimesteps)
    {
        return new idg::GridderPlan(idg::Type::CPU_REFERENCE, bufferTimesteps);
    }




    void GridderPlan_destroy(idg::GridderPlan* p) {
       delete p;
    }

} // extern C
