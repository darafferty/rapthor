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
          m_nrStations(0),
          m_nrGroups(0),
          m_lastTimeIndex(-1),
          m_wOffsetInMeters(0.0f),
          m_nrPolarizations(4),
          m_wKernelSize(0),
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


    // deprecated: use cell size!
    void GridderPlan::set_image_size(const double imageSize)
    {
        m_imageSize = float(imageSize);
    }


    // deprecated: use cell size!
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


    size_t GridderPlan::get_w_kernel_size() const
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
        flush();
    }


    void GridderPlan::set_grid(
        std::complex<double>* grid,
        const size_t nr_polarizations,
        const size_t size)
    {
        set_grid(grid, nr_polarizations, size, size);
    }


    void GridderPlan::set_grid(
        std::complex<double>* grid,
        const size_t nr_polarizations,
        const size_t height,
        const size_t width)
    {
        m_grid_double = grid;

        // For later support of non-square grids and nr_polarizations=1
        if (height != width)
            throw invalid_argument("Only square grids supported.");
        if (nr_polarizations != 4)
            throw invalid_argument("The number of polarization paris must be equals 4.");

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
        // params.set_nr_polarizations(m_nrPolarizations);

        #if defined(BUILD_LIB_CPU)
        if (m_architecture == Type::CPU_REFERENCE)
            m_proxy = new proxy::cpu::Reference(params);
        else if (m_architecture == Type::CPU_OPTIMIZED)
            m_proxy = new proxy::cpu::HaswellEP(params);
        else
            throw invalid_argument("Unknown architecture type.");
        #endif

        // (2) Setup buffers
        malloc_gridder_buffers();
    }


    void GridderPlan::malloc_gridder_buffers()
    {
        m_bufferUVW.reserve(m_nrGroups, m_bufferTimesteps);
        m_bufferVisibilities.reserve(m_nrGroups, m_bufferTimesteps, get_frequencies_size());
        m_bufferStationPairs.reserve(m_nrGroups);
        m_grid.reserve(m_nrPolarizations, m_gridHeight, m_gridWidth);
        m_aterms.reserve(m_nrStations, m_subgridSize, m_subgridSize);

        // HACK:
        for (auto s = 0; s < m_nrStations; ++s)
            for (auto y = 0; y < m_subgridSize; ++y)
                for (auto x = 0; x < m_subgridSize; ++x)
                    m_aterms(s, y, x) = {complex<float>(1), complex<float>(0),
                                         complex<float>(0), complex<float>(1)};
    }


    void GridderPlan::free_gridder_buffers()
    {
        // to be implemented
        // m_bufferUVW.free();
        // m_bufferVisibilities.free();
        // m_bufferStationPairs.free();
        // m_grid.free();
        // m_aterms.free();
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
        auto local_time = timeIndex - m_lastTimeIndex - 1;
        auto local_bl = baseline_index(antenna1, antenna2);

        if (local_time == m_bufferTimesteps) {
            /* Do not insert more if buffer is already full */
            /* Execute and empty buffer, befor inserting new element */
            flush();
            local_time = timeIndex - m_lastTimeIndex - 1;
        } else {
            // Keep track of all time indices pushed into the buffer
            m_timeindices.insert(timeIndex);

            // Copy data into buffers
            m_bufferUVW(local_bl, local_time) = {
                static_cast<float>(uvwInMeters[0]),
                static_cast<float>(uvwInMeters[1]),
                static_cast<float>(uvwInMeters[2])
            };

            if (antenna1 > antenna2) swap(antenna1, antenna2);
            m_bufferStationPairs[local_bl] = {
                static_cast<int>(antenna1),
                static_cast<int>(antenna2)
            };

            copy(visibilities, visibilities + get_frequencies_size() * m_nrPolarizations,
                 (complex<float>*) &m_bufferVisibilities(local_bl, local_time, 0));
        }
    }


    void GridderPlan::start_aterm(
        const std::complex<double>* aterm,
        const size_t nrStations,
        const size_t height,
        const size_t width,
        const size_t nrPolarizations)
    {
        if (nrStations != m_nrStations)
            throw invalid_argument("The number of stations to not match the plan.");
        if (nrPolarizations != nrPolarizations)
            throw invalid_argument("The number of polarization to not match the plan.");

        // to be implemented
        // TODO: remove hack to ignore aterm
        // WARNING: layout does not match the one in the kernel
        // TODO: Resize to SUBGRIDSIZE x SUBGRIDSIZE on the fly
        for (auto s = 0; s < m_nrStations; ++s)
            for (auto y = 0; y < m_subgridSize; ++y)
                for (auto x = 0; x < m_subgridSize; ++x) {
                    size_t ind = s*height*width*nrPolarizations +
                                 y*width*nrPolarizations
                        + x*nrPolarizations;
                    m_aterms(s, y, x) = {complex<float>(aterm[ind + 0]),
                                         complex<float>(aterm[ind + 1]),
                                         complex<float>(aterm[ind + 2]),
                                         complex<float>(aterm[ind + 3])};
                }
    }


    void GridderPlan::start_aterm(
        const std::complex<double>* aterm,
        const size_t nrStations,
        const size_t size,
        const size_t nrPolarizations)
    {
        start_aterm(aterm, nrStations, size, size, nrPolarizations);
    }


    void GridderPlan::finish_aterm()
    {
        // to be implemented
        flush();
    }



    // Must be called whenever the buffer is full or no more data added
    void GridderPlan::flush()
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
        for (auto p = 0; p < m_nrPolarizations; ++p) {
            for (auto y = 0; y < m_gridHeight; ++y) {
                for (auto x = 0; x < m_gridWidth; ++x) {
                    m_grid_double[p*m_gridHeight*m_gridWidth
                                  + y*m_gridWidth + x] += m_grid(p, y, x);
                }
            }
        }

        // Cleanup
        auto largestTimeIndex = *max_element( m_timeindices.cbegin(), m_timeindices.cend() );
        m_lastTimeIndex = largestTimeIndex;
        m_timeindices.clear();
        // init buffers to zero
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


    int GridderPlan_get_stations(idg::GridderPlan* p)
    {
        return p->get_stations();
    }


    void GridderPlan_set_stations(idg::GridderPlan* p, int n) {
        p->set_stations(n);
    }


    void GridderPlan_set_frequencies(
        idg::GridderPlan* p,
        double* frequencyList,
        int size)
    {
        p->set_frequencies(frequencyList, size);
    }


    double GridderPlan_get_frequency(idg::GridderPlan* p, int channel)
    {
        return p->get_frequency(channel);
    }


    int GridderPlan_get_frequencies_size(idg::GridderPlan* p)
    {
        return p->get_frequencies_size();
    }


    void GridderPlan_set_w_kernel_size(idg::GridderPlan* p, int size)
    {
        p->set_w_kernel(size);
    }


    int GridderPlan_get_w_kernel_size(idg::GridderPlan* p)
    {
        return p->get_w_kernel_size();
    }


    void GridderPlan_set_subgrid_size(idg::GridderPlan* p, int size)
    {
        p->set_subgrid_size(size);
    }


    int GridderPlan_get_subgrid_size(idg::GridderPlan* p)
    {
        return p->get_subgrid_size();
    }


    void GridderPlan_set_grid(
        idg::GridderPlan* p,
        void* grid,   // ptr to complex double
        int nr_polarizations,
        int height,
        int width
        )
    {
        p->set_grid(
            (std::complex<double>*) grid,
            nr_polarizations,
            height,
            width);
    }


    void GridderPlan_set_spheroidal(
        idg::GridderPlan* p,
        double* spheroidal,
        int height,
        int width)
    {
        p->set_spheroidal(spheroidal, height, width);
    }



    // deprecated: use cell size!
    void GridderPlan_set_image_size(idg::GridderPlan* p, double imageSize)
    {
        p->set_image_size(imageSize);
    }


    // deprecated: use cell size!
    double GridderPlan_get_image_size(idg::GridderPlan* p)
    {
        return p->get_image_size();
    }


    void GridderPlan_bake(idg::GridderPlan* p)
    {
        p->bake();
    }


    void GridderPlan_start_aterm(
        idg::GridderPlan* p,
        void* aterm,  // ptr to complex double
        int nrStations,
        int height,
        int width,
        int nrPolarizations)
    {
        p->start_aterm(
            (std::complex<double>*) aterm,
            nrStations,
            height,
            width,
            nrPolarizations);
    }


    void GridderPlan_finish_aterm(idg::GridderPlan* p)
    {
        p->finish_aterm();
    }


    void GridderPlan_grid_visibilities(
        idg::GridderPlan* p,
        float*  visibilities, // size CH x PL x 2
        double* uvwInMeters,
        int     antenna1,
        int     antenna2,
        int     timeIndex)
    {
        p->grid_visibilities(
            (complex<float>*) visibilities, // size CH x PL
            uvwInMeters,
            antenna1,
            antenna2,
            timeIndex);
    }


    void GridderPlan_flush(idg::GridderPlan* p)
    {
        p->flush();
    }


    void GridderPlan_destroy(idg::GridderPlan* p) {
       delete p;
    }

} // extern C
