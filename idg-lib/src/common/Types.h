#ifndef IDG_TYPES_H_
#define IDG_TYPES_H_

#include <ostream>

namespace idg {

    /* Structures */

    typedef struct { float u, v, w; } UVW; // TODO: remove

    typedef struct { int x, y; } Coordinate;

    typedef struct { int station1, station2; } Baseline;

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
                m_delete_buffer(true),
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
                for (unsigned int i = 0; i < get_x_dim(); ++i) {
                    (*this)(i) = a;
                }
            }

        private:
            const size_t m_x_dim;
            const bool   m_delete_buffer;
            T*           m_buffer;
    };


    template<class T>
    class Array2D {
        public:
            Array2D(
                size_t height,
                size_t width) :
                m_x_dim(width),
                m_y_dim(height),
                m_delete_buffer(true),
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

            virtual ~Array2D()
            {
                if (m_delete_buffer) delete[] m_buffer;
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
                for (unsigned int y = 0; y < get_y_dim(); ++y) {
                    for (unsigned int x = 0; x < get_x_dim(); ++x) {
                        (*this)(y, x) = a;
                    }
                }
            }

        private:
            const size_t m_x_dim;
            const size_t m_y_dim;
            const bool   m_delete_buffer;
            T*           m_buffer;
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
                m_delete_buffer(true),
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
                for (unsigned int z = 0; z < get_z_dim(); ++z) {
                    for (unsigned int y = 0; y < get_y_dim(); ++y) {
                        for (unsigned int x = 0; x < get_x_dim(); ++x) {
                            (*this)(z, y, x) = a;
                        }
                    }

                }
            }

        private:
            const size_t m_x_dim;
            const size_t m_y_dim;
            const size_t m_z_dim;
            const bool   m_delete_buffer;
            T*           m_buffer;
    };


    template<class T>
    class Array4D {
        public:
            Array4D(
                size_t w_dim,
                size_t z_dim,
                size_t y_dim,
                size_t x_dim) :
                m_w_dim(w_dim),
                m_z_dim(z_dim),
                m_y_dim(y_dim),
                m_x_dim(x_dim),
                m_delete_buffer(true),
                m_buffer(new T[w_dim*z_dim*y_dim*x_dim])
            {}

            Array4D(
                T* data,
                size_t w_dim,
                size_t z_dim,
                size_t y_dim,
                size_t x_dim) :
                m_w_dim(w_dim),
                m_z_dim(z_dim),
                m_y_dim(y_dim),
                m_x_dim(x_dim),
                m_delete_buffer(false),
                m_buffer(data)
            {}

            Array4D(const Array4D& other) = delete;
            Array4D& operator=(const Array4D& rhs) = delete;

            virtual ~Array4D() { if (m_delete_buffer) delete[] m_buffer; }

            T* data(
                size_t w=0,
                size_t z=0,
                size_t y=0,
                size_t x=0) const
            {
                return &m_buffer[x + m_x_dim*y + m_x_dim*m_y_dim*z + m_x_dim*m_y_dim*m_z_dim*w];
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
                return m_buffer[x + m_x_dim*y + m_x_dim*m_y_dim*z + m_x_dim*m_y_dim*m_z_dim*w];
            }

            T& operator()(
                size_t w,
                size_t z,
                size_t y,
                size_t x)
            {
                return m_buffer[x + m_x_dim*y + m_x_dim*m_y_dim*z + m_x_dim*m_y_dim*m_z_dim*w];
            }

            void init(const T& a) {
                for (unsigned int w = 0; z < get_w_dim(); ++w) {
                    for (unsigned int z = 0; z < get_z_dim(); ++z) {
                        for (unsigned int y = 0; y < get_y_dim(); ++y) {
                            for (unsigned int x = 0; x < get_x_dim(); ++x) {
                                (*this)(w, z, y, x) = a;
                            }
                        }
                    }
                }
            }

        private:
            const size_t m_x_dim;
            const size_t m_y_dim;
            const size_t m_z_dim;
            const size_t m_w_dim;
            const bool   m_delete_buffer;
            T*           m_buffer;
    };


    /* Output */

    std::ostream& operator<<(std::ostream& os, Baseline& b);
    std::ostream& operator<<(std::ostream& os, Coordinate& c);
    std::ostream& operator<<(std::ostream& os, Metadata& m);
    std::ostream& operator<<(std::ostream& os, UVW& uvw);

    std::ostream& operator<<(std::ostream& os, const float2& x);
    std::ostream& operator<<(std::ostream& os, const double2& x);
}


#endif
