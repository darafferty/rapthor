/*
 * Scheme.h
 * Access to IDG's high level gridder routines
 */


#include "Scheme.h"

using namespace std;

namespace idg {

    // Constructors and destructor
    Scheme::Scheme(Type architecture,
                   size_t bufferTimesteps)
        : m_architecture(architecture),
          m_bufferTimesteps(bufferTimesteps),
          m_timeStartThisBatch(0),
          m_timeStartNextBatch(bufferTimesteps),
          m_nrStations(0),
          m_nrGroups(0),
          m_wOffsetInLambda(0.0f),
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

    Scheme::~Scheme()
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif
    }


    // Set/get all parameters

    void Scheme::set_stations(const size_t nrStations)
    {
        m_nrStations = nrStations;
        m_nrGroups = ((nrStations - 1) * nrStations) / 2;
    }


    size_t Scheme::get_stations() const
    {
        return m_nrStations;
    }


    // deprecated: use cell size!
    void Scheme::set_image_size(const double imageSize)
    {
        m_imageSize = float(imageSize);
    }


    // deprecated: use cell size!
    double Scheme::get_image_size() const
    {
        return m_imageSize;
    }


    void Scheme::set_cell_size(const double cellSize)
    {
        m_cellSize = float(cellSize);
    }


    double Scheme::get_cell_size() const
    {
        return m_cellSize;
    }


    void Scheme::internal_set_subgrid_size(const size_t size)
    {
        m_subgridSize = size;
    }


    size_t Scheme::internal_get_subgrid_size() const
    {
        return m_subgridSize;
    }


    void Scheme::set_w_kernel(size_t size)
    {
        m_wKernelSize = size;
    }


    size_t Scheme::get_w_kernel_size() const
    {
        return m_wKernelSize;
    }


    void Scheme::set_spheroidal(
        size_t size,
        const double* spheroidal)
    {
        set_spheroidal(size, size, spheroidal);
    }


    void Scheme::set_spheroidal(
        size_t height,
        size_t width,
        const double* spheroidal)
    {
        // TODO: first, resize to SUBGRIDSIZE x SUBGRIDSIZE
        m_spheroidal.reserve(height, width);
        for (auto y = 0; y < height; ++y) {
            for (auto x = 0; x < width; x++) {
                m_spheroidal(y, x) = float(spheroidal[y*width + x]);
            }
        }
    }


    void Scheme::set_frequencies(
        size_t channelCount,
        const double* frequencyList)
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


    double Scheme::get_frequency(const size_t channel) const
    {
        return m_frequencies[channel];
    }


    size_t Scheme::get_frequencies_size() const
    {
        return m_frequencies.size();
    }


    void Scheme::start_w_layer(double wOffsetInLambda)
    {
        m_wOffsetInLambda = float(wOffsetInLambda);
    }


    void Scheme::finish_w_layer()
    {
        flush();
    }


    void Scheme::set_grid(
        const size_t nr_polarizations,
        const size_t size,
        std::complex<double>* grid)
    {
        set_grid(nr_polarizations, size, size, grid);
    }


    void Scheme::set_grid(
        const size_t nr_polarizations,
        const size_t height,
        const size_t width,
        std::complex<double>* grid)
    {
        // For later support of non-square grids and nr_polarizations=1
        if (height != width)
            throw invalid_argument("Only square grids supported.");
        if (nr_polarizations != 4)
            throw invalid_argument("The number of polarization pairs must be equals 4.");

        m_nrPolarizations = nr_polarizations;
        m_gridHeight      = height;
        m_gridWidth       = width;
        m_grid_double     = grid;
    }


    size_t Scheme::get_nr_polarizations() const
    {
        return m_nrPolarizations;
    }


    size_t Scheme::get_grid_height() const
    {
        return m_gridHeight;
    }


    size_t Scheme::get_grid_width() const
    {
        return m_gridWidth;
    }


    // Plan creation and helper functions

    void Scheme::bake()
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif

        // HACK: assume that, if image size not set, cell size is
        // NOTE: assume m_gridWidth == m_gridHeight
        // TODO: remove image size from this api entirely
        if (m_imageSize==0) m_imageSize = m_cellSize * m_gridWidth;

        // (1) Create new proxy
        delete m_proxy;

        Parameters params;
        params.set_nr_stations(m_nrStations);
        params.set_nr_time(m_bufferTimesteps);
        params.set_nr_timeslots(1);
        params.set_nr_channels(get_frequencies_size()); // TODO: remove as compile time const
        params.set_grid_size(m_gridHeight);             // TODO: support non-square
        params.set_subgrid_size(m_subgridSize);
        params.set_imagesize(m_imageSize); // TODO: remove as compile time const
        // params.set_nr_polarizations(m_nrPolarizations);

        #if defined(BUILD_LIB_CPU)
        if (m_architecture == Type::CPU_REFERENCE) {
            m_proxy = new proxy::cpu::Reference(params);
        } else if (m_architecture == Type::CPU_OPTIMIZED) {
            m_proxy = new proxy::cpu::HaswellEP(params);
        }
        #endif

        if (m_proxy == nullptr)
            throw invalid_argument("Unknown architecture type.");

        // (2) Setup buffers
        malloc_buffers();
        reset_buffers(); // optimization: only call "set_uvw_to_infinity()" here
    }


    void Scheme::malloc_buffers()
    {
        m_bufferUVW.reserve(m_nrGroups, m_bufferTimesteps);
        m_bufferVisibilities.reserve(m_nrGroups, m_bufferTimesteps, get_frequencies_size());
        m_bufferStationPairs.resize(m_nrGroups);
        m_grid.reserve(m_nrPolarizations, m_gridHeight, m_gridWidth);
        m_spheroidal.reserve(m_subgridSize, m_subgridSize);
        m_aterms.reserve(m_nrStations, m_subgridSize, m_subgridSize);
    }


    void Scheme::reset_buffers()
    {
        m_bufferVisibilities.init({0,0,0,0});
        set_uvw_to_infinity();
        init_default_aterm();
    }


    void Scheme::set_uvw_to_infinity()
    {
        m_bufferUVW.init({numeric_limits<float>::infinity(),
                          numeric_limits<float>::infinity(),
                          numeric_limits<float>::infinity()});
    }


    void Scheme::init_default_aterm() {
        for (auto s = 0; s < m_nrStations; ++s)
            for (auto y = 0; y < m_subgridSize; ++y)
                for (auto x = 0; x < m_subgridSize; ++x)
                    m_aterms(s, y, x) = {complex<float>(1), complex<float>(0),
                                         complex<float>(0), complex<float>(1)};
    }


    /* The baseline index is formed such that:
     *   0 implies antenna1=0, antenna2=1 ;
     *   1 implies antenna1=0, antenna2=2 ;
     * n-1 implies antenna1=1, antenna2=2 etc. */
    size_t Scheme::baseline_index(size_t antenna1, size_t antenna2) const
    {
        if (antenna1 > antenna2) {
            swap(antenna1, antenna2);
        }
        auto offset =  antenna1*m_nrStations - ((antenna1+1)*antenna1)/2 - 1;
        return antenna2 - antenna1 + offset;
    }


    void Scheme::start_aterm(
        size_t nrStations,
        size_t height,
        size_t width,
        size_t nrPolarizations,
        const std::complex<double>* aterm)
    {
        if (nrStations != m_nrStations)
            throw invalid_argument("The number of stations to not match the plan.");
        if (nrPolarizations != m_nrPolarizations)
            throw invalid_argument("The number of polarization to not match the plan.");

        // to be implemented
        // TODO: remove hack to ignore aterm
        // WARNING: layout does not match the one in the kernel
        // TODO: Resize to SUBGRIDSIZE x SUBGRIDSIZE on the fly
        for (auto s = 0; s < m_nrStations; ++s)
            for (auto y = 0; y < m_subgridSize; ++y)
                for (auto x = 0; x < m_subgridSize; ++x) {
                    size_t ind = s*height*width*nrPolarizations +
                                 y*width*nrPolarizations + x*nrPolarizations;
                    m_aterms(s, y, x) = {complex<float>(aterm[ind + 0]),
                                         complex<float>(aterm[ind + 1]),
                                         complex<float>(aterm[ind + 2]),
                                         complex<float>(aterm[ind + 3])};
                }
    }


    void Scheme::start_aterm(
        size_t nrStations,
        size_t size,
        size_t nrPolarizations,
        const std::complex<double>* aterm)
    {
        start_aterm(nrStations, size, size, nrPolarizations, aterm);
    }


    void Scheme::finish_aterm()
    {
        flush();
    }


    void Scheme::fft_kernel(
        Direction direction,
        size_t    nr_polarizations,
        size_t    height,
        size_t    width,
        complex<double> *grid)
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif

        #pragma omp parallel for
        for (int pol = 0; pol < m_nrPolarizations; pol++) {
            if (direction == Direction::FourierToImage) {
                fft2(height, width, &grid[pol*height*width]);
            } else {
                ifft2(height, width, &grid[pol*height*width]);
            }
        }
    }


    void Scheme::fft_grid(
        size_t nr_polarizations,
        size_t height,
        size_t width,
        complex<double> *grid)
    {
        // Normal case: no arguments -> transform member grid
        // Note: the other case is to perform the transform on a copy
        // so that the process can be monitored
        if (grid == nullptr) {
            nr_polarizations = m_nrPolarizations;
            height           = m_gridHeight;
            width            = m_gridWidth;
            grid             = m_grid_double;
        }

        fft_kernel(Direction::ImageToFourier, nr_polarizations, height, width, grid);
    }


    void Scheme::ifft_grid(
        size_t nr_polarizations,
        size_t height,
        size_t width,
        complex<double> *grid)
    {
        // Normal case: no arguments -> transform member grid
        // Note: the other case is to perform the transform on a copy
        // so that the process can be monitored
        if (grid == nullptr) {
            nr_polarizations = m_nrPolarizations;
            height           = m_gridHeight;
            width            = m_gridWidth;
            grid             = m_grid_double;
        }

        fft_kernel(Direction::FourierToImage, nr_polarizations, height, width, grid);
    }


    void Scheme::copy_grid(
        size_t nr_polarizations,
        size_t height,
        size_t width,
        complex<double>* grid)
    {
        if (nr_polarizations != m_nrPolarizations)
            throw invalid_argument("Number of polarizations does not match.");
        if (height != m_gridHeight)
            throw invalid_argument("Grid height does not match.");
        if (width != m_gridWidth)
            throw invalid_argument("Grid width does not match.");

        for (auto p = 0; p < m_nrPolarizations; ++p) {
            for (auto y = 0; y < m_gridHeight; ++y) {
                for (auto x = 0; x < m_gridWidth; ++x) {
                    grid[p*m_gridHeight*m_gridWidth +
                         y*m_gridWidth + x] =
                    m_grid_double[p*m_gridHeight*m_gridWidth +
                                  y*m_gridWidth + x];

                }
            }
        }
    }

} // namespace idg




// C interface:
// Rationale: calling the code from C code and Fortran easier,
// and bases to create interface to scripting languages such as
// Python, Julia, Matlab, ...
extern "C" {

    int Scheme_get_stations(idg::Scheme* p)
    {
        return p->get_stations();
    }


    void Scheme_set_stations(idg::Scheme* p, int n) {
        p->set_stations(n);
    }


    void Scheme_set_frequencies(
        idg::Scheme* p,
        int nr_channels,
        double* frequencies)
    {
        p->set_frequencies(nr_channels, frequencies);
    }


    double Scheme_get_frequency(idg::Scheme* p, int channel)
    {
        return p->get_frequency(channel);
    }


    int Scheme_get_frequencies_size(idg::Scheme* p)
    {
        return p->get_frequencies_size();
    }


    void Scheme_set_w_kernel_size(idg::Scheme* p, int size)
    {
        p->set_w_kernel(size);
    }


    int Scheme_get_w_kernel_size(idg::Scheme* p)
    {
        return p->get_w_kernel_size();
    }


    void Scheme_set_grid(
        idg::Scheme* p,
        int nr_polarizations,
        int height,
        int width,
        void* grid)   // ptr to complex double
    {
        p->set_grid(
            nr_polarizations,
            height,
            width,
            (std::complex<double>*) grid);
    }


    int Scheme_get_nr_polarizations(idg::Scheme* p)
    {
        return p->get_nr_polarizations();
    }


    int Scheme_get_grid_height(idg::Scheme* p)
    {
        return p->get_grid_height();
    }


    int Scheme_get_grid_width(idg::Scheme* p)
    {
        return p->get_grid_width();
    }


    void Scheme_set_spheroidal(
        idg::Scheme* p,
        int height,
        int width,
        double* spheroidal)
    {
        p->set_spheroidal(height, width, spheroidal);
    }


    // deprecated: use cell size!
    void Scheme_set_image_size(idg::Scheme* p, double imageSize)
    {
        p->set_image_size(imageSize);
    }


    // deprecated: use cell size!
    double Scheme_get_image_size(idg::Scheme* p)
    {
        return p->get_image_size();
    }


    void Scheme_bake(idg::Scheme* p)
    {
        p->bake();
    }


    void Scheme_start_aterm(
        idg::Scheme* p,
        int nrStations,
        int height,
        int width,
        int nrPolarizations,
        void* aterm)  // ptr to complex double
    {
        p->start_aterm(
            nrStations,
            height,
            width,
            nrPolarizations,
            (std::complex<double>*) aterm);
    }


    void Scheme_finish_aterm(idg::Scheme* p)
    {
        p->finish_aterm();
    }


    void Scheme_flush(idg::Scheme* p)
    {
        p->flush();
    }


    void Scheme_internal_set_subgrid_size(idg::Scheme* p, int size)
    {
        p->internal_set_subgrid_size(size);
    }


    int Scheme_internal_get_subgrid_size(idg::Scheme* p)
    {
        return p->internal_get_subgrid_size();
    }


    void Scheme_ifft_grid(idg::Scheme* p)
    {
        p->ifft_grid();
    }


    void Scheme_fft_grid(idg::Scheme* p)
    {
        p->fft_grid();
    }


    void Scheme_copy_grid(
        idg::Scheme* p,
        int   nr_polarizations,
        int   height,
        int   width,
        void* grid)   // pointer complex<double>
    {
        p->copy_grid(
            nr_polarizations,
            height,
            width,
            (complex<double>*) grid);
    }

} // extern C
