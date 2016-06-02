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
        const double* spheroidal,
        const size_t size)
    {
        set_spheroidal(spheroidal, size, size);
    }


    void Scheme::set_spheroidal(
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


    void Scheme::set_frequencies(
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
        std::complex<double>* grid,
        const size_t nr_polarizations,
        const size_t size)
    {
        set_grid(grid, nr_polarizations, size, size);
    }


    void Scheme::set_grid(
        std::complex<double>* grid,
        const size_t nr_polarizations,
        const size_t height,
        const size_t width)
    {
        // For later support of non-square grids and nr_polarizations=1
        if (height != width)
            throw invalid_argument("Only square grids supported.");
        if (nr_polarizations != 4)
            throw invalid_argument("The number of polarization pairs must be equals 4.");

        m_grid_double     = grid;
        m_nrPolarizations = nr_polarizations;
        m_gridHeight      = height;
        m_gridWidth       = width;
    }


    size_t Scheme::get_grid_height() const
    {
        return m_gridHeight;
    }

    size_t Scheme::get_grid_width() const
    {
        return m_gridWidth;
    }


    size_t Scheme::get_nr_polarizations() const
    {
        return m_nrPolarizations;
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
        const std::complex<double>* aterm,
        const size_t nrStations,
        const size_t height,
        const size_t width,
        const size_t nrPolarizations)
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
        const std::complex<double>* aterm,
        const size_t nrStations,
        const size_t size,
        const size_t nrPolarizations)
    {
        start_aterm(aterm, nrStations, size, size, nrPolarizations);
    }


    void Scheme::finish_aterm()
    {
        flush();
    }


    void Scheme::fft(Direction direction, complex<double> *grid)
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif

        if (grid == nullptr) grid = m_grid_double;

        // HACK:: copy array
        Grid3D<complex<float>> tmp_grid(m_nrPolarizations, m_gridHeight, m_gridWidth);
        for (auto p = 0; p < m_nrPolarizations; ++p) {
            for (auto y = 0; y < m_gridHeight; ++y) {
                for (auto x = 0; x < m_gridWidth; ++x) {
                    tmp_grid(p, y, x) = complex<float>(
                        grid[p*m_gridHeight*m_gridWidth +
                             y*m_gridWidth + x]);
                }
            }
        }

        if (direction == Direction::FourierToImage)
            ifftshift(m_nrPolarizations, tmp_grid.data()); // TODO: integrate into adder?
        else
            ifftshift(m_nrPolarizations, tmp_grid.data()); // TODO: remove

        auto sign = (direction == Direction::ImageToFourier) ? FFTW_FORWARD : FFTW_BACKWARD;

        #pragma omp parallel for
        for (int pol = 0; pol < m_nrPolarizations; pol++) {
            // fftwf_complex *data = (fftwf_complex *) tmp_grid.data(pol);
            fftwf_complex *data = (fftwf_complex *) tmp_grid.data() +
                                  pol * (m_gridHeight * m_gridWidth);

            // Create plan
            fftwf_plan plan;
            #pragma omp critical
            plan = fftwf_plan_dft_2d(m_gridHeight, m_gridWidth, // TODO: order
                                     data, data,
                                     sign, FFTW_ESTIMATE);

            // Execute FFTs
            fftwf_execute_dft(plan, data, data);

            // Scaling in case of an inverse FFT, so that FFT(iFFT())=identity()
            if (sign == FFTW_BACKWARD) {
                float scale = 1.0f / (float(m_gridHeight)*float(m_gridWidth));
                #pragma omp parallel for
                for (int i = 0; i < m_gridHeight*m_gridWidth; i++) {
                    data[i][0] *= scale;
                    data[i][1] *= scale;
                }
            }

            // Destroy plan
            fftwf_destroy_plan(plan);
        }

        if (direction == Direction::FourierToImage)
            fftshift(m_nrPolarizations, tmp_grid.data()); // TODO: remove
        else
            fftshift(m_nrPolarizations, tmp_grid.data()); // TODO: integrate into splitter?

        // HACK:: copy array
        for (auto p = 0; p < m_nrPolarizations; ++p) {
            for (auto y = 0; y < m_gridHeight; ++y) {
                for (auto x = 0; x < m_gridWidth; ++x) {
                    grid[p*m_gridHeight*m_gridWidth +
                         y*m_gridWidth + x] =
                        complex<double>(tmp_grid(p, y, x));
                }
            }
        }
    }


    void Scheme::fft_grid(complex<double> *grid)
    {
        fft(Direction::ImageToFourier, grid);
    }


    void Scheme::ifft_grid(complex<double> *grid)
    {
        fft(Direction::FourierToImage, grid);
    }


    void Scheme::copy_grid(
        complex<double>* grid,
        size_t nr_polarizations,
        size_t height,
        size_t width)
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


    void Scheme::ifftshift(int nr_polarizations, complex<float> *grid)
    {
        // TODO: implement for odd size gridsize
        // For even gridsize, same as fftshift
        fftshift(nr_polarizations, grid);
    }


    void Scheme::fftshift(int nr_polarizations, complex<float> *grid)
    {
        // Note: grid[NR_POLARIZATIONS][GRIDSIZE][GRIDSIZE]
        auto gridsize = get_grid_height();

        #pragma omp parallel for
        for (int p = 0; p < get_nr_polarizations(); p++) {
            // Pass &grid[p][GRIDSIZE][GRIDSIZE]
            // and do shift for each polarization
            fftshift(grid + p*gridsize*gridsize);
        }
    }


    void Scheme::ifftshift(complex<float> *array)
    {
        // TODO: implement
    }


    void Scheme::fftshift(complex<float> *array)
    {
        auto gridsize = get_grid_height();
        auto buffer   = new complex<float>[gridsize];

        if (gridsize % 2 != 0)
            throw invalid_argument("gridsize is assumed to be even");

        for (int i = 0; i < gridsize/2; i++) {
            // save i-th row into buffer
            memcpy(buffer, &array[i*gridsize],
                   gridsize*sizeof(complex<float>));

            auto j = i + gridsize/2;
            memcpy(&array[i*gridsize + gridsize/2], &array[j*gridsize],
                   (gridsize/2)*sizeof(complex<float>));
            memcpy(&array[i*gridsize], &array[j*gridsize + gridsize/2],
                   (gridsize/2)*sizeof(complex<float>));
            memcpy(&array[j*gridsize], &buffer[gridsize/2],
                   (gridsize/2)*sizeof(complex<float>));
            memcpy(&array[j*gridsize + gridsize/2], &buffer[0],
                   (gridsize/2)*sizeof(complex<float>));
        }

        delete [] buffer;
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
        double* frequencyList,
        int size)
    {
        p->set_frequencies(frequencyList, size);
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
        double* spheroidal,
        int height,
        int width)
    {
        p->set_spheroidal(spheroidal, height, width);
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
        void* grid,   // pointer complex<double>
        int   nr_polarizations,
        int   height,
        int   width)
    {
        p->copy_grid(
            (complex<double>*) grid,
            nr_polarizations,
            height,
            width);
    }

    // void Scheme_transform_grid(
    //     idg::Scheme* p,
    //     int direction)
    // {
    //     if (direction == 0) {
    //         p->transform_grid(idg::Direction::FourierToImage,
    //                           (std::complex<double>*) grid,
    //                           nr_polarizations,
    //                           height, width);
    //     } else {
    //         p->transform_grid(idg::Direction::ImageToFourier,
    //                           (std::complex<double>*) grid,
    //                           nr_polarizations,
    //                           height, width);
    //     }
    // }


} // extern C
