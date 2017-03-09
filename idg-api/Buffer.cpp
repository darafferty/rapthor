/*
 * BufferImpl.cpp
 * Access to IDG's high level gridder routines
 */

#include "BufferImpl.h"

using namespace std;

namespace idg {
namespace api {

    // Constructors and destructor
    BufferImpl::BufferImpl(Type architecture,
                   size_t bufferTimesteps)
        : m_architecture(architecture),
          m_bufferTimesteps(bufferTimesteps),
          m_timeStartThisBatch(0),
          m_timeStartNextBatch(bufferTimesteps),
          m_nrStations(0),
          m_nrGroups(0),
          m_nrPolarizations(4),
          m_gridHeight(0),
          m_gridWidth(0),
          m_subgridSize(0),
          m_wStepInLambda(0.0f),
          m_cellHeight(0.0f),
          m_cellWidth(0.0f),
          m_kernel_size(0),
          m_aterm_offsets(2),
          m_frequencies(0),
          m_wavenumbers(100),
          m_spheroidal(0,0),                            
          m_aterms(0,0,0,0),            
          m_bufferUVW(0,0),
          m_bufferStationPairs(0),
          m_bufferVisibilities(0,0,0),
          m_proxy(nullptr)
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif
        m_aterm_offsets(0) = 0;
        m_aterm_offsets(1) = bufferTimesteps;
    }

    BufferImpl::~BufferImpl()
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif
        delete m_proxy;
    }

    // Set/get all parameters

    void BufferImpl::set_stations(const size_t nrStations)
    {
        m_nrStations = nrStations;
        m_nrGroups = ((nrStations - 1) * nrStations) / 2;
    }


    size_t BufferImpl::get_stations() const
    {
        return m_nrStations;
    }

    double BufferImpl::get_image_size() const
    {
        return m_cellWidth * m_gridWidth;
    }


    void BufferImpl::set_cell_size(double height, double width)
    {
        if (height != width)
            throw invalid_argument("Only square cells supported.");

        m_cellHeight = float(height);
        m_cellWidth  = float(width);
    }


    double BufferImpl::get_cell_height() const
    {
        return m_cellHeight;
    }


    double BufferImpl::get_cell_width() const
    {
        return m_cellWidth;
    }


    void BufferImpl::set_w_step(float w_step)
    {
        m_wStepInLambda = w_step;
    }


    float BufferImpl::get_w_step() const
    {
        return m_wStepInLambda;
    }

    void BufferImpl::set_subgrid_size(const size_t size)
    {
        m_subgridSize = size;
    }


    size_t BufferImpl::get_subgrid_size() const
    {
        return m_subgridSize;
    }


    void BufferImpl::set_kernel_size(float size)
    {
        m_kernel_size = size;
    }


    float BufferImpl::get_kernel_size() const
    {
        return m_kernel_size;
    }


    void BufferImpl::set_spheroidal(
        size_t size,
        const float* spheroidal)
    {
        set_spheroidal(size, size, spheroidal);
    }


    void BufferImpl::set_spheroidal(
        size_t height,
        size_t width,
        const float* spheroidal)
    {
        // TODO: first, resize to SUBGRIDSIZE x SUBGRIDSIZE
        std::cout << height << "," << width << std::endl;
        m_spheroidal = Array2D<float>(height, width);
        for (auto y = 0; y < height; ++y) {
            for (auto x = 0; x < width; x++) {
                m_spheroidal(y, x) = float(spheroidal[y*width + x]);
            }
        }
    }


    void BufferImpl::set_frequencies(
        size_t channelCount,
        const double* frequencyList)
    {
        const double SPEED_OF_LIGHT = 299792458.0;
        m_frequencies = Array1D<float>(channelCount);
        m_wavenumbers = Array1D<float>(channelCount);
        for (int i=0; i<channelCount; i++) {
            m_frequencies(i) = frequencyList[i];
            m_wavenumbers(i) = 2 * M_PI * frequencyList[i] / SPEED_OF_LIGHT;
        }
    }

    void BufferImpl::set_frequencies(
        const std::vector<double> &frequency_list)
    {
        const int channelCount = frequency_list.size();
        const double SPEED_OF_LIGHT = 299792458.0;
        m_frequencies = Array1D<float>(channelCount);
        m_wavenumbers = Array1D<float>(channelCount);
        for (int i=0; i<channelCount; i++) {
            m_frequencies(i) = frequency_list[i];
            m_wavenumbers(i) = 2 * M_PI * frequency_list[i] / SPEED_OF_LIGHT;
        }
    }

    double BufferImpl::get_frequency(const size_t channel) const
    {
        return m_frequencies(channel);
    }


    size_t BufferImpl::get_frequencies_size() const
    {
        return m_frequencies.get_x_dim();
    }

    void BufferImpl::set_grid(
        Grid* grid)
    {
        m_grid            = grid;
    }


    size_t BufferImpl::get_nr_polarizations() const
    {
        return m_nrPolarizations;
    }


    size_t BufferImpl::get_grid_height() const
    {
        return m_gridHeight;
    }


    size_t BufferImpl::get_grid_width() const
    {
        return m_gridWidth;
    }


    // Plan creation and helper functions

    void BufferImpl::bake()
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif

        // HACK: assume that, if image size not set, cell size is
        // NOTE: assume m_gridWidth == m_gridHeight

        // (1) Create new proxy
        delete m_proxy;

        int nr_correlations = 4;
        CompileConstants constants(nr_correlations, m_subgridSize);

        #if defined(BUILD_LIB_CPU)
        if (m_architecture == Type::CPU_REFERENCE) {
            m_proxy = new proxy::cpu::Reference(constants);
        } else if (m_architecture == Type::CPU_OPTIMIZED) {
            m_proxy = new proxy::cpu::Optimized(constants);
        }
        #endif
        #if defined(BUILD_LIB_CUDA)
        if (m_architecture == Type::CUDA_GENERIC) {
            m_proxy = new proxy::cuda::Generic(constants);
        }
        #endif
        #if defined(BUILD_LIB_CPU) && defined(BUILD_LIB_CUDA)
        if (m_architecture == Type::HYBRID_CUDA_CPU_OPTIMIZED) {
            // cpu proxy will be deleted by hybrid proxy destructor
            proxy::cpu::CPU *cpu_proxy = new proxy::cpu::Optimized(constants);
            m_proxy = new proxy::hybrid::HybridCUDA(cpu_proxy, constants);
        }
        #endif
        #if defined(BUILD_LIB_OPENCL)
        if (m_architecture == Type::OPENCL_GENERIC) {
            m_proxy = new proxy::opencl::Generic(constants);
        }
        #endif

        if (m_proxy == nullptr)
            throw invalid_argument("Unknown architecture type.");

        // (2) Setup buffers
        malloc_buffers();
        reset_buffers(); // optimization: only call "set_uvw_to_infinity()" here
    }


    void BufferImpl::malloc_buffers()
    {
        m_bufferUVW = Array2D<UVWCoordinate<float>>(m_nrGroups, m_bufferTimesteps);
        m_bufferVisibilities = Array3D<Visibility<std::complex<float>>>(m_nrGroups, m_bufferTimesteps, get_frequencies_size());
        m_bufferStationPairs = Array1D<std::pair<unsigned int,unsigned int>>(m_nrGroups);
        // already done: m_spheroidal.reserve(m_subgridSize, m_subgridSize);
        m_aterms = Array4D<Matrix2x2<std::complex<float>>>(1, m_nrStations, m_subgridSize, m_subgridSize);
    }


    void BufferImpl::reset_buffers()
    {
        m_bufferVisibilities.init({0,0,0,0});
        set_uvw_to_infinity();
        init_default_aterm();
    }


    void BufferImpl::set_uvw_to_infinity()
    {
        m_bufferUVW.init({numeric_limits<float>::infinity(),
                          numeric_limits<float>::infinity(),
                          numeric_limits<float>::infinity()});
    }


    void BufferImpl::init_default_aterm() {
        for (auto s = 0; s < m_nrStations; ++s)
            for (auto y = 0; y < m_subgridSize; ++y)
                for (auto x = 0; x < m_subgridSize; ++x)
                    m_aterms(0, s, y, x) = {complex<float>(1), complex<float>(0),
                                         complex<float>(0), complex<float>(1)};
    }


    /* The baseline index is formed such that:
     *   0 implies antenna1=0, antenna2=1 ;
     *   1 implies antenna1=0, antenna2=2 ;
     * n-1 implies antenna1=1, antenna2=2 etc. */
    size_t BufferImpl::baseline_index(size_t antenna1, size_t antenna2) const
    {
        assert(antenna1 < antenna2);
        auto offset =  antenna1*m_nrStations - ((antenna1+1)*antenna1)/2 - 1;
        return antenna2 - antenna1 + offset;
    }


    void BufferImpl::start_aterm(
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
                    m_aterms(0, s, y, x) = {complex<float>(aterm[ind + 0]),
                                         complex<float>(aterm[ind + 1]),
                                         complex<float>(aterm[ind + 2]),
                                         complex<float>(aterm[ind + 3])};
                }
    }


    void BufferImpl::start_aterm(
        size_t nrStations,
        size_t size,
        size_t nrPolarizations,
        const std::complex<double>* aterm)
    {
        start_aterm(nrStations, size, size, nrPolarizations, aterm);
    }


    void BufferImpl::finish_aterm()
    {
        flush();
    }


    void BufferImpl::fft_grid(
        size_t nr_polarizations,
        size_t height,
        size_t width,
        complex<float> *grid)
    {
        #pragma omp parallel for
        for (int pol = 0; pol < m_nrPolarizations; pol++) {
            fftshift(height, width, &grid[pol*height*width]); // TODO: remove shift here
            fft2f(height, width, &grid[pol*height*width]);
        }
    }


    void BufferImpl::ifft_grid(
        size_t nr_polarizations,
        size_t height,
        size_t width,
        complex<float> *grid)
    {
//         // Normal case: no arguments -> transform member grid
//         // Note: the other case is to perform the transform on a copy
//         // so that the process can be monitored
//         if (grid == nullptr) {
//             nr_polarizations = m_nrPolarizations;
//             height           = m_gridHeight;
//             width            = m_gridWidth;
//             grid             = m_grid_double;
//         }
// 
        #pragma omp parallel for
        for (int pol = 0; pol < m_nrPolarizations; pol++) {
            ifft2f(height, width, &grid[pol*height*width]);
            fftshift(height, width, &grid[pol*height*width]); // TODO: remove shift here
        }
    }


    void BufferImpl::copy_grid(
        size_t nr_polarizations,
        size_t height,
        size_t width,
        complex<double>* grid)
    {
//         if (nr_polarizations != m_nrPolarizations)
//             throw invalid_argument("Number of polarizations does not match.");
//         if (height != m_gridHeight)
//             throw invalid_argument("Grid height does not match.");
//         if (width != m_gridWidth)
//             throw invalid_argument("Grid width does not match.");
// 
//         for (auto p = 0; p < m_nrPolarizations; ++p) {
//             for (auto y = 0; y < m_gridHeight; ++y) {
//                 for (auto x = 0; x < m_gridWidth; ++x) {
//                     grid[p*m_gridHeight*m_gridWidth +
//                          y*m_gridWidth + x] =
//                     m_grid_double[p*m_gridHeight*m_gridWidth +
//                                   y*m_gridWidth + x];
// 
//                 }
//             }
//         }
    }

} // namespace api
} // namespace idg




// C interface:
// Rationale: calling the code from C code and Fortran easier,
// and bases to create interface to scripting languages such as
// Python, Julia, Matlab, ...
extern "C" {

    int Buffer_get_stations(idg::api::BufferImpl* p)
    {
        return p->get_stations();
    }


    void Buffer_set_stations(idg::api::BufferImpl* p, int n) {
        p->set_stations(n);
    }


    void Buffer_set_frequencies(
        idg::api::BufferImpl* p,
        int nr_channels,
        double* frequencies)
    {
        p->set_frequencies(nr_channels, frequencies);
    }


    double Buffer_get_frequency(idg::api::BufferImpl* p, int channel)
    {
        return p->get_frequency(channel);
    }


    int Buffer_get_frequencies_size(idg::api::BufferImpl* p)
    {
        return p->get_frequencies_size();
    }


    void Buffer_set_w_kernel_size(idg::api::BufferImpl* p, int size)
    {
        p->set_kernel_size(size);
    }


    int Buffer_get_w_kernel_size(idg::api::BufferImpl* p)
    {
        return p->get_kernel_size();
    }


//     void Buffer_set_grid(
//         idg::api::Buffer* p,
//         int nr_polarizations,
//         int height,
//         int width,
//         void* grid)   // ptr to complex double
//     {
//         p->set_grid(
//             nr_polarizations,
//             height,
//             width,
//             (std::complex<float>*) grid);
//     }
// 

    int Buffer_get_nr_polarizations(idg::api::BufferImpl* p)
    {
        return p->get_nr_polarizations();
    }


    int Buffer_get_grid_height(idg::api::BufferImpl* p)
    {
        return p->get_grid_height();
    }


    int Buffer_get_grid_width(idg::api::BufferImpl* p)
    {
        return p->get_grid_width();
    }


    void Buffer_set_spheroidal(
        idg::api::BufferImpl* p,
        int height,
        int width,
        float* spheroidal)
    {
        p->set_spheroidal(height, width, spheroidal);
    }


    void Buffer_set_cell_size(idg::api::BufferImpl* p, double height, double width)
    {
        p->set_cell_size(height, width);
    }


    double Buffer_get_cell_height(idg::api::BufferImpl* p)
    {
        return p->get_cell_height();
    }


    double Buffer_get_cell_width(idg::api::BufferImpl* p)
    {
        return p->get_cell_width();
    }

    double Buffer_get_image_size(idg::api::BufferImpl* p)
    {
        return p->get_image_size();
    }


    void Buffer_bake(idg::api::BufferImpl* p)
    {
        p->bake();
    }


    void Buffer_start_aterm(
        idg::api::BufferImpl* p,
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


    void Buffer_finish_aterm(idg::api::BufferImpl* p)
    {
        p->finish_aterm();
    }


    void Buffer_flush(idg::api::BufferImpl* p)
    {
        p->flush();
    }


    void Buffer_set_subgrid_size(idg::api::BufferImpl* p, int size)
    {
        p->set_subgrid_size(size);
    }


    int Buffer_get_subgrid_size(idg::api::BufferImpl* p)
    {
        return p->get_subgrid_size();
    }


    void Buffer_ifft_grid(idg::api::BufferImpl* p)
    {
        p->ifft_grid();
    }


    void Buffer_fft_grid(idg::api::BufferImpl* p)
    {
        p->fft_grid();
    }


    void Buffer_copy_grid(
        idg::api::BufferImpl* p,
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
