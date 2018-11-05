#ifndef IDG_TYPES_H_
#define IDG_TYPES_H_

#include <ostream>
#include <complex>
#include <cassert>
#include <cstring>
#include <memory>

namespace idg {

    /* Structures */
    typedef struct { int x, y, z; } Coordinate;

    typedef struct { unsigned int station1, station2; } Baseline;

    typedef struct {
        int baseline_offset;
        int time_offset;
        int nr_timesteps;
        int aterm_index;
        Baseline baseline;
        Coordinate coordinate; } Metadata;

    typedef struct {float real; float imag; } float2;
    typedef struct {double real; double imag; } double2;

    template<class T>
    struct Matrix2x2 {T xx; T xy; T yx; T yy;};

    template<class T>
    using Visibility = Matrix2x2<T>;

    template<class T>
    struct UVWCoordinate {T u; T v; T w;};


    /* Inline operations */
    inline float2 operator*(const float2& x, const float2& y)
    {
        return {x.real*y.real - x.imag*y.imag,
                x.real*y.imag + x.imag*y.real};
    }

    inline float2 operator*(const float2& x, const float& a)
    {
        return {x.real*a, x.imag*a};
    }

    inline float2 operator*(const float& a, const float2& x)
    {
        return x*a;
    }

    inline float2 operator+(const float2& x, const float2& y)
    {
        return {x.real + y.real, x.imag + y.imag};
    }

    inline void operator+=(float2& x, const float2& y) {
        x.real += y.real;
        x.imag += y.imag;
    }

    inline float2 conj(const float2& x) {
        return {x.real, -x.imag};
    }

    inline double2 operator*(const double2& x, const double2& y)
    {
        return {x.real*y.real - x.imag*y.imag,
                x.real*y.imag + x.imag*y.real};
    }

    inline double2 operator*(const double2& x, const double& a)
    {
        return {x.real*a, x.imag*a};
    }

    inline double2 operator*(const double& a, const double2& x)
    {
        return x*a;
    }

    inline double2 operator+(const double2& x, const double2& y)
    {
        return {x.real + y.real, x.imag + y.imag};
    }

    inline void operator+=(double2& x, const double2& y) {
        x.real += y.real;
        x.imag += y.imag;
    }

    inline double2 conj(const double2& x) {
        return {x.real, -x.imag};
    }


    /* Classes */
    template<class T>
    class Array1D {
        public:
            Array1D(
                size_t width) :
                m_x_dim(width),
                m_delete_buffer(width > 0),
                m_buffer(new T[width])
            {}

            Array1D(
                T* data,
                size_t width) :
                m_x_dim(width),
                m_delete_buffer(false),
                m_buffer(data)
            {}

            Array1D(const Array1D& v) = delete;
            Array1D& operator=(const Array1D& rhs) = delete;

            Array1D(Array1D&& other)
                : m_x_dim(other.m_x_dim),
                  m_delete_buffer(other.m_delete_buffer),
                  m_buffer(other.m_buffer)
            {
                other.m_buffer = nullptr;
            }

            Array1D& operator=(Array1D&& other)
            {
                if (m_delete_buffer) delete[] m_buffer;
                m_x_dim = other.m_x_dim;
                m_delete_buffer = other.m_delete_buffer;
                m_buffer = other.m_buffer;
                other.m_buffer = nullptr;
				return *this;
            }

            virtual ~Array1D()
            {
                if (m_delete_buffer) delete[] m_buffer;
            }

            T* data(
                size_t index=0) const
            {
                return &m_buffer[index];
            }

            size_t get_x_dim() const { return m_x_dim; }

            const T& operator()(
                size_t i) const
            {
                return m_buffer[i];
            }

            T& operator()(
                size_t i)
            {
                return m_buffer[i];
            }

            void init(const T& a) {
                const unsigned int n = m_x_dim;
                for (unsigned int i = 0; i < n; ++i) {
                    m_buffer[i] = a;
                }
            }

            size_t bytes() const {
                return get_x_dim() * sizeof(T);
            }

        protected:
            size_t m_x_dim;
            bool   m_delete_buffer;
            T*     m_buffer;
    };


    template<class T>
    class Array2D {
        public:
            Array2D(
                size_t height,
                size_t width) :
                m_x_dim(width),
                m_y_dim(height),
                m_delete_buffer((height*width) > 0),
                m_buffer(new T[height*width])
            {}

            Array2D(
                T* data,
                size_t height,
                size_t width) :
                m_x_dim(width),
                m_y_dim(height),
                m_delete_buffer(false),
                m_buffer(data)
            {}

            Array2D(const Array2D& v) = delete;
            Array2D& operator=(const Array2D& rhs) = delete;

            Array2D(Array2D&& other)
                : m_x_dim(other.m_x_dim),
                  m_y_dim(other.m_y_dim),
                  m_delete_buffer(other.m_delete_buffer),
                  m_buffer(other.m_buffer)
            {
                other.m_buffer = nullptr;
            }

            // move assignment operator
            Array2D& operator=(Array2D&& other)
            {
                if (m_delete_buffer) delete[] m_buffer;
                m_x_dim = other.m_x_dim;
                m_y_dim = other.m_y_dim;
                m_delete_buffer = other.m_delete_buffer;
                m_buffer = other.m_buffer;
                other.m_buffer = nullptr;
				return *this;
            }

            virtual ~Array2D()
            {
                if (m_delete_buffer) {
                    delete[] m_buffer;
                }
            }

            T* data(
                size_t row=0,
                size_t column=0) const
            {
                return &m_buffer[row*m_x_dim + column];
            }

            size_t get_x_dim() const { return m_x_dim; }
            size_t get_y_dim() const { return m_y_dim; }

            const T& operator()(
                size_t y,
                size_t x) const
            {
                return m_buffer[x + m_x_dim*y];
            }

            T& operator()(
                size_t y,
                size_t x)
            {
                return m_buffer[x + m_x_dim*y];
            }

            void init(const T& a) {
                const unsigned int n = m_x_dim*m_y_dim;
                for (unsigned int i = 0; i < n; ++i) {
                    m_buffer[i] = a;
                }
            }

            size_t bytes() const {
                return get_y_dim() *
                       get_x_dim() * sizeof(T);
            }

        protected:
            size_t m_x_dim;
            size_t m_y_dim;
            bool   m_delete_buffer;
            T*     m_buffer;
    };


    template<class T>
    class Array3D {
        public:
            Array3D(
                size_t depth,
                size_t height,
                size_t width) :
                m_x_dim(width),
                m_y_dim(height),
                m_z_dim(depth),
                m_delete_buffer((width*height*depth) > 0),
                m_buffer(new T[height*width*depth])
            {}

            Array3D(
                T* data,
                size_t depth,
                size_t height,
                size_t width) :
                m_x_dim(width),
                m_y_dim(height),
                m_z_dim(depth),
                m_delete_buffer(false),
                m_buffer(data)
            {}

            Array3D(const Array3D& other) = delete;
            Array3D& operator=(const Array3D& rhs) = delete;

            Array3D(Array3D&& other)
                : m_x_dim(other.m_x_dim),
                  m_y_dim(other.m_y_dim),
                  m_z_dim(other.m_z_dim),
                  m_delete_buffer(other.m_delete_buffer),
                  m_buffer(other.m_buffer)
            {
                other.m_buffer = nullptr;
            }

            // move assignment operator
            Array3D& operator=(Array3D&& other)
            {
                if (m_delete_buffer) delete[] m_buffer;
                m_x_dim = other.m_x_dim;
                m_y_dim = other.m_y_dim;
                m_z_dim = other.m_z_dim;
                m_delete_buffer = other.m_delete_buffer;
                m_buffer = other.m_buffer;
                other.m_buffer = nullptr;
				return *this;
            }

            virtual ~Array3D() { if (m_delete_buffer) delete[] m_buffer; }

            T* data(
                size_t z=0,
                size_t y=0,
                size_t x=0) const
            {
                return &m_buffer[x + m_x_dim*y + m_x_dim*m_y_dim*z];
            }

            size_t get_x_dim() const { return m_x_dim; }
            size_t get_y_dim() const { return m_y_dim; }
            size_t get_z_dim() const { return m_z_dim; }

            const T& operator()(
                size_t z,
                size_t y,
                size_t x) const
            {
                return m_buffer[x + m_x_dim*y + m_x_dim*m_y_dim*z];
            }

            T& operator()(
                size_t z,
                size_t y,
                size_t x)
            {
                return m_buffer[x + m_x_dim*y + m_x_dim*m_y_dim*z];
            }

            void init(const T& a) {
                const unsigned int n = m_x_dim*m_y_dim*m_z_dim;
                for (unsigned int i = 0; i < n; ++i) {
                    m_buffer[i] = a;
                }
            }

            size_t bytes() const {
                return get_z_dim() * get_y_dim() *
                       get_x_dim() * sizeof(T);
            }

        protected:
            size_t m_x_dim;
            size_t m_y_dim;
            size_t m_z_dim;
            bool   m_delete_buffer;
            T*     m_buffer;
    };


    template<class T>
    class Array4D {
        public:
            Array4D(
                size_t w_dim,
                size_t z_dim,
                size_t y_dim,
                size_t x_dim) :
                m_x_dim(x_dim),
                m_y_dim(y_dim),
                m_z_dim(z_dim),
                m_w_dim(w_dim),
                m_buffer( new T[w_dim*z_dim*y_dim*x_dim], std::default_delete<T[]>() )  // shared_ptr with custom deleter that deletes an array
            {}

            Array4D(
                std::shared_ptr<T> data,
                size_t w_dim,
                size_t z_dim,
                size_t y_dim,
                size_t x_dim) :
                m_x_dim(x_dim),
                m_y_dim(y_dim),
                m_z_dim(z_dim),
                m_w_dim(w_dim),
                m_buffer(data)
            {}

            Array4D(
                T* data,
                size_t w_dim,
                size_t z_dim,
                size_t y_dim,
                size_t x_dim) :
                m_x_dim(x_dim),
                m_y_dim(y_dim),
                m_z_dim(z_dim),
                m_w_dim(w_dim),
                m_buffer(data, [](T*){}) // shared_ptr with custom deleter that does nothing
            {}

            Array4D(const Array4D& other) = delete;
            Array4D& operator=(const Array4D& rhs) = delete;

            Array4D(Array4D&& other)
                : m_x_dim(other.m_x_dim),
                  m_y_dim(other.m_y_dim),
                  m_z_dim(other.m_z_dim),
                  m_w_dim(other.m_w_dim),
                  m_buffer(other.m_buffer)
            {
                other.m_buffer = nullptr;
            }

            // move assignment operator
            Array4D& operator=(Array4D&& other)
            {
                m_w_dim = other.m_w_dim;
                m_x_dim = other.m_x_dim;
                m_y_dim = other.m_y_dim;
                m_z_dim = other.m_z_dim;
                m_buffer = other.m_buffer;
                other.m_buffer = nullptr;
				return *this;
            }

            virtual ~Array4D() {}

            T* data(
                size_t w=0,
                size_t z=0,
                size_t y=0,
                size_t x=0) const
            {
                return &m_buffer.get()[x + m_x_dim*y + m_x_dim*m_y_dim*z + m_x_dim*m_y_dim*m_z_dim*w];
            }

            size_t get_x_dim() const { return m_x_dim; }
            size_t get_y_dim() const { return m_y_dim; }
            size_t get_z_dim() const { return m_z_dim; }
            size_t get_w_dim() const { return m_w_dim; }

            const T& operator()(
                size_t w,
                size_t z,
                size_t y,
                size_t x) const
            {
                return m_buffer.get()[x + m_x_dim*y + m_x_dim*m_y_dim*z + m_x_dim*m_y_dim*m_z_dim*w];
            }

            T& operator()(
                size_t w,
                size_t z,
                size_t y,
                size_t x)
            {
                return m_buffer.get()[x + m_x_dim*y + m_x_dim*m_y_dim*z + m_x_dim*m_y_dim*m_z_dim*w];
            }

            void init(const T& a) {
                const unsigned int n = m_x_dim*m_y_dim*m_z_dim*m_w_dim;
                for (unsigned int i = 0; i < n; ++i) {
                    m_buffer.get()[i] = a;
                }
            }

            size_t bytes() const {
                return get_w_dim() * get_z_dim() *
                       get_y_dim() * get_x_dim() * sizeof(T);
            }

            // TODO: if the buffer is not owned, there is no guarantee that it won't be destroyed.
            // Need to return a copy in that case.
            const std::shared_ptr<const T> get() const {return m_buffer;}

        protected:
            size_t m_x_dim;
            size_t m_y_dim;
            size_t m_z_dim;
            size_t m_w_dim;
            std::shared_ptr<T> m_buffer;
    };

    class Grid : public Array4D<std::complex<float>> {
        public:
            Grid(Array3D<std::complex<float>> &array) :
                    Array4D<std::complex<float>>(array.data(),
                                                 1,
                                                 array.get_z_dim(),
                                                 array.get_y_dim(),
                                                 array.get_x_dim())
                {}

                Grid(std::complex<float>* data,
                    size_t w_dim,
                    size_t z_dim,
                    size_t y_dim,
                    size_t x_dim) :
                    Array4D<std::complex<float>>(data, w_dim, z_dim, y_dim, x_dim)
                {}

                Grid(
                    size_t w_dim,
                    size_t z_dim,
                    size_t y_dim,
                    size_t x_dim) :
                    Array4D<std::complex<float>>(w_dim, z_dim, y_dim, x_dim)
                {}

                void zero() {
                    memset((void *) m_buffer.get(), 0, bytes());
                }
    };


    /* Output */
    std::ostream& operator<<(std::ostream& os, Baseline& b);
    std::ostream& operator<<(std::ostream& os, Coordinate& c);
    std::ostream& operator<<(std::ostream& os, Metadata& m);

    template<class T>
    std::ostream& operator<<(std::ostream& os, UVWCoordinate<T>& uvw);

    std::ostream& operator<<(std::ostream& os, const float2& x);
    std::ostream& operator<<(std::ostream& os, const double2& x);

    template<class T>
    std::ostream& operator<<(
        std::ostream& os,
        const Array1D<T>& a)
    {
        for (unsigned int x = 0; x < a.get_x_dim(); ++x) {
            os << a(x);
            if (x != a.get_x_dim()-1) {
                os << ",";
            }
        }
        os << std::endl;
        return os;
    }

    template<class T>
    std::ostream& operator<<(
        std::ostream& os,
        const Array2D<T>& a)
    {
        for (unsigned int y = 0; y < a.get_y_dim(); ++y) {
            for (unsigned int x = 0; x < a.get_x_dim(); ++x) {
                os << a(y,x);
                if (x != a.get_x_dim()-1) {
                    os << ",";
                }
            }
            os << std::endl;
        }
        return os;
    }

    template<class T>
    std::ostream& operator<<(
        std::ostream& os,
        const Array3D<T>& a)
    {
        for (unsigned int z = 0; z < a.get_z_dim(); ++z) {
            os << std::endl;
            for (unsigned int y = 0; y < a.get_y_dim(); ++y) {
                for (unsigned int x = 0; x < a.get_x_dim(); ++x) {
                    os << a(z,y,x);
                    if (x != a.get_x_dim()-1) {
                        os << ",";
                    }
                }
                os << std::endl;
            }
        }
        return os;
    }

    template<class T>
    std::ostream& operator<<(
        std::ostream& os,
        const Array4D<T>& a)
    {
        for (unsigned int w = 0; w < a.get_w_dim(); ++w) {
            os << std::endl;
            for (unsigned int z = 0; z < a.get_z_dim(); ++z) {
                os << std::endl;
                for (unsigned int y = 0; y < a.get_y_dim(); ++y) {
                    for (unsigned int x = 0; x < a.get_x_dim(); ++x) {
                        os << a(w, z,y,x);
                        if (x != a.get_x_dim()-1) {
                            os << ",";
                        }
                    }
                    os << std::endl;
                }
            }
        }
        return os;
    }

} // end namespace idg

    /* Index methods */
    inline size_t index_grid(
            int nr_polarizations,
            long grid_size,
            int w_layer,
            int pol,
            int y,
            int x)
    {
        // grid: [nr_w_layers][nr_polarizations][grid_size][grid_size]
        return static_cast<size_t>(w_layer) * nr_polarizations * grid_size * grid_size +
               static_cast<size_t>(pol) * grid_size * grid_size +
               static_cast<size_t>(y) * grid_size +
               static_cast<size_t>(x);
    }

    inline size_t index_grid_tiling(
            int tile_size,
            int nr_correlations,
            size_t grid_size,
            int pol,
            int y,
            int x)
    {
        // grid: [NR_TILES][NR_TILES][NR_POLARIZATIONS][TILE_SIZE][TILE_SIZE]
        assert(grid_size % tile_size == 0);
        const int NR_TILES  = grid_size / tile_size;
        size_t idx_tile_y = y / tile_size;
        size_t idx_tile_x = x / tile_size;
        size_t tile_y = y % tile_size;
        size_t tile_x = x % tile_size;

        return
               idx_tile_y * NR_TILES * nr_correlations * tile_size * tile_size +
               idx_tile_x * nr_correlations * tile_size * tile_size +
               pol * tile_size * tile_size +
               tile_y * tile_size +
               tile_x;
    }

    inline size_t index_grid(
            size_t grid_size,
            int pol,
            int y,
            int x)
    {
        // grid: [nr_polarizations][grid_size][grid_size]
        return pol * grid_size * grid_size +
               y * grid_size +
               x;
    }

    inline size_t index_subgrid(
        int nr_polarizations,
        int subgrid_size,
        int s,
        int pol,
        int y,
        int x)
    {
        // subgrid: [nr_subgrids][nr_polarizations][subgrid_size][subgrid_size]
        return static_cast<size_t>(s) * nr_polarizations * subgrid_size * subgrid_size +
               static_cast<size_t>(pol) * subgrid_size * subgrid_size +
               static_cast<size_t>(y) * subgrid_size +
               static_cast<size_t>(x);
    }

    inline size_t index_visibility(
        int nr_channels,
        int nr_polarizations,
        int time,
        int chan,
        int pol)
    {
        // visibilities: [nr_time][nr_channels][nr_polarizations]
        return static_cast<size_t>(time) * nr_channels * nr_polarizations +
               static_cast<size_t>(chan) * nr_polarizations +
               static_cast<size_t>(pol);
    }

    inline size_t index_aterm(
        int subgrid_size,
        int nr_polarizations,
        int nr_stations,
        int aterm_index,
        int station,
        int y,
        int x)
    {
        // aterm: [nr_aterms][subgrid_size][subgrid_size][nr_polarizations]
        size_t aterm_nr = (aterm_index * nr_stations + station);
        return static_cast<size_t>(aterm_nr) * subgrid_size * subgrid_size * nr_polarizations +
               static_cast<size_t>(y) * subgrid_size * nr_polarizations +
               static_cast<size_t>(x) * nr_polarizations;
    }

#endif
