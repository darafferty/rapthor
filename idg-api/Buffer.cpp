/*
 * BufferImpl.cpp
 * Access to IDG's high level gridder routines
 */

#include "BufferImpl.h"
#include "BufferSetImpl.h"

#include <csignal>


using namespace std;

namespace idg {
namespace api {

    inline float meters_to_pixels(float meters, float imagesize, float frequency) {
        const double speed_of_light = 299792458.0;
        return meters * imagesize * (frequency / speed_of_light);
    }



    // Constructors and destructor
    BufferImpl::BufferImpl(
        BufferSetImpl* bufferset,
        proxy::Proxy* proxy,
        size_t bufferTimesteps)
        : m_bufferset(bufferset),
          m_max_baseline(0.0),
          m_uv_span_frequency(0.0),
          m_bufferTimesteps(bufferTimesteps),
          m_timeStartThisBatch(0),
          m_timeStartNextBatch(bufferTimesteps),
          m_nrStations(0),
          m_nr_baselines(0),
          m_nrPolarizations(4),
          m_gridHeight(0),
          m_gridWidth(0),
          m_subgridsize(bufferset->m_subgridsize),
          m_wStepInLambda(0.0f),
          m_shift(3),
          m_cellHeight(0.0f),
          m_cellWidth(0.0f),
          m_kernel_size(0),
          m_default_aterm_offsets(2),
          m_aterm_offsets_array(0),
          m_frequencies(0),
          m_spheroidal(0,0),
          m_aterms_array(0,0,0,0),
          m_bufferUVW(0,0),
          m_bufferStationPairs(0),
          m_bufferVisibilities(0, 0, 0),
          m_proxy(proxy)
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif
        m_default_aterm_offsets[0] = 0;
        m_default_aterm_offsets[1] = bufferTimesteps;
        m_aterm_offsets = m_default_aterm_offsets;
    }

    BufferImpl::~BufferImpl()
    {
        #if defined(DEBUG)
        cout << __func__ << endl;
        #endif
    }

    // Set/get all parameters

    void BufferImpl::set_stations(const size_t nrStations)
    {
        m_nrStations = nrStations;
        m_nr_baselines = ((nrStations - 1) * nrStations) / 2;
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

    void BufferImpl::set_shift(const float* shift)
    {
        m_shift(0) = shift[0];
        m_shift(1) = shift[1];
        m_shift(2) = shift[2];
    }

    const idg::Array1D<float>& BufferImpl::get_shift() const
    {
        return m_shift;
    }

    void BufferImpl::set_subgrid_size(const size_t size)
    {
        m_subgridsize = size;
    }


    size_t BufferImpl::get_subgrid_size() const
    {
        return m_subgridsize;
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
#ifndef NDEBUG
        std::cout << height << "," << width << std::endl;
#endif
        m_spheroidal = m_proxy->allocate_array2d<float>(height, width);
        for (auto y = 0; y < height; ++y) {
            for (auto x = 0; x < width; x++) {
                m_spheroidal(y, x) = float(spheroidal[y*width + x]);
            }
        }
    }


    void BufferImpl::set_frequencies(
        size_t nr_channels,
        const double* frequencyList)
    {
        m_nr_channels = nr_channels;
        m_frequencies = m_proxy->allocate_array1d<float>(m_nr_channels);
        for (int i=0; i<m_nr_channels; i++) {
            m_frequencies(i) = frequencyList[i];
        }
    }

    void BufferImpl::set_frequencies(
        const std::vector<double> &frequency_list)
    {
        m_nr_channels = frequency_list.size();
        m_frequencies = m_proxy->allocate_array1d<float>(m_nr_channels);
        for (int i=0; i<m_nr_channels; i++) {
            m_frequencies(i) = frequency_list[i];
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
        std::shared_ptr<Grid> grid)
    {
        m_grid = grid;
        m_gridHeight  = grid->get_x_dim();
        m_gridWidth   = grid->get_y_dim();
        m_nr_w_layers = grid->get_w_dim();
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

        // NOTE: assume m_gridWidth == m_gridHeight

        // Setup buffers
        malloc_buffers();
        reset_buffers(); // optimization: only call "set_uvw_to_infinity()" here
    }


    void BufferImpl::malloc_buffers()
    {
        m_bufferUVW = m_proxy->allocate_array2d<UVW<float>>(m_nr_baselines, m_bufferTimesteps);
        m_bufferVisibilities = m_proxy->allocate_array3d<Visibility<std::complex<float>>>(m_nr_baselines, m_bufferTimesteps, m_nr_channels);
        m_bufferStationPairs = m_proxy->allocate_array1d<std::pair<unsigned int,unsigned int>>(m_nr_baselines);
        m_bufferStationPairs.init({m_nrStations, m_nrStations});
        // already done: m_spheroidal.reserve(m_subgridsize, m_subgridsize);
        // m_aterms = Array4D<Matrix2x2<std::complex<float>>>(1, m_nrStations, m_subgridsize, m_subgridsize);
    }


    void BufferImpl::reset_buffers()
    {
        m_bufferVisibilities.zero();
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
        m_default_aterms.resize(1*m_nrStations*m_subgridsize*m_subgridsize);
        for (auto s = 0; s < m_nrStations; ++s) {
            for (auto y = 0; y < m_subgridsize; ++y) {
                for (auto x = 0; x < m_subgridsize; ++x) {
                    size_t offset = m_subgridsize*m_subgridsize*s +
                                    m_subgridsize*y + x;
                    m_default_aterms[offset] = {complex<float>(1), complex<float>(0),
                                                complex<float>(0), complex<float>(1)};
                }
            }
        }
        m_aterms = m_default_aterms;
    }

    // Set the a-term that starts validity at timeIndex
    void BufferImpl::set_aterm(
        size_t timeIndex,
        const complex<float>* aterms)
    {
        int n_ants = m_nrStations;
        int local_time = timeIndex - m_timeStartThisBatch;
        size_t n_old_aterms = m_aterm_offsets.size()-1;
        size_t atermBlockSize = m_nrStations*m_subgridsize*m_subgridsize;

        // Overwrite last a-term if new timeindex same as one but last element aterm_offsets
        if (local_time != m_aterm_offsets[m_aterm_offsets.size()-2]) {
          assert(local_time > m_aterm_offsets[m_aterm_offsets.size()-2]);
          assert(local_time <= m_bufferTimesteps);

          // insert new timeIndex before the last element in m_aterm_offsets
          assert(m_aterm_offsets.size() == n_old_aterms+1);
          m_aterm_offsets.resize(n_old_aterms+2);
          m_aterm_offsets[n_old_aterms+2-1] = m_bufferTimesteps;
          m_aterm_offsets[n_old_aterms+2-2] = local_time;
          // push back new a-term
          m_aterms.resize(m_aterms.size()+atermBlockSize);
          assert(m_aterms.size()==(n_old_aterms+1)*atermBlockSize);
        }
        size_t n_new_aterms = m_aterm_offsets.size()-1;
        std::copy(aterms,
                  aterms + atermBlockSize*4,
                  (complex<float>*) (m_aterms.data()+(n_new_aterms-1)*atermBlockSize));
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
