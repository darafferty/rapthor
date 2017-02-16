#ifndef IDG_TYPES_H_
#define IDG_TYPES_H_

#include <ostream>
#include <complex>

namespace idg {

    /* Structures */
    
    typedef struct { int x, y; } Coordinate;

    typedef struct { unsigned int station1, station2; } Baseline;

    typedef struct {
        int baseline_offset;
        int time_offset;
        int nr_timesteps;
        int aterm_index;
        int w_index;
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
                m_buffer((T*) malloc(width*sizeof(T)))
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
                if (m_delete_buffer) free(m_buffer);
                m_x_dim = other.m_x_dim;
                m_delete_buffer = other.m_delete_buffer;
                m_buffer = other.m_buffer;
                other.m_buffer = nullptr;
            }

            virtual ~Array1D()
            {
                if (m_delete_buffer) free(m_buffer);
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

            size_t bytes() const {
                return get_x_dim() * sizeof(T);
            }

        private:
            size_t m_x_dim;
            bool   m_delete_buffer;
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
                m_delete_buffer((height*width) > 0),
                m_buffer((T*) malloc(height*width*sizeof(T)))
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
                if (m_delete_buffer) free(m_buffer);
                m_x_dim = other.m_x_dim;
                m_y_dim = other.m_y_dim;
                m_delete_buffer = other.m_delete_buffer;
                m_buffer = other.m_buffer;
                other.m_buffer = nullptr;
            }

            virtual ~Array2D()
            {
                if (m_delete_buffer) {
                    free(m_buffer);
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
                for (unsigned int y = 0; y < get_y_dim(); ++y) {
                    for (unsigned int x = 0; x < get_x_dim(); ++x) {
                        (*this)(y, x) = a;
                    }
                }
            }

            size_t bytes() const {
                return get_y_dim() *
                       get_x_dim() * sizeof(T);
            }

        private:
            size_t m_x_dim;
            size_t m_y_dim;
            bool   m_delete_buffer;
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
                m_delete_buffer((width*height*depth) > 0),
                m_buffer((T*) malloc(height*width*depth*sizeof(T)))
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
                if (m_delete_buffer) free(m_buffer);
                m_x_dim = other.m_x_dim;
                m_y_dim = other.m_y_dim;
                m_z_dim = other.m_z_dim;
                m_delete_buffer = other.m_delete_buffer;
                m_buffer = other.m_buffer;
                other.m_buffer = nullptr;
            }

            virtual ~Array3D() { if (m_delete_buffer) free(m_buffer); }

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

            size_t bytes() const {
                return get_z_dim() * get_y_dim() *
                       get_x_dim() * sizeof(T);
            }

        private:
            size_t m_x_dim;
            size_t m_y_dim;
            size_t m_z_dim;
            bool   m_delete_buffer;
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
                m_delete_buffer((w_dim*x_dim*y_dim*z_dim) > 0),
                m_buffer((T*) malloc(w_dim*z_dim*y_dim*x_dim*sizeof(T)))
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

            Array4D(Array4D&& other)
                : m_x_dim(other.m_x_dim),
                  m_y_dim(other.m_y_dim),
                  m_z_dim(other.m_z_dim),
                  m_w_dim(other.m_w_dim),
                  m_delete_buffer(other.m_delete_buffer),
                  m_buffer(other.m_buffer)
            {
                other.m_buffer = nullptr;
            }

            // move assignment operator
            Array4D& operator=(Array4D&& other)
            {
                if (m_delete_buffer) free(m_buffer);
                m_w_dim = other.m_w_dim;
                m_x_dim = other.m_x_dim;
                m_y_dim = other.m_y_dim;
                m_z_dim = other.m_z_dim;
                m_delete_buffer = other.m_delete_buffer;
                m_buffer = other.m_buffer;
                other.m_buffer = nullptr;
            }

            virtual ~Array4D() { if (m_delete_buffer) free(m_buffer); }

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
                for (unsigned int w = 0; w < get_w_dim(); ++w) {
                    for (unsigned int z = 0; z < get_z_dim(); ++z) {
                        for (unsigned int y = 0; y < get_y_dim(); ++y) {
                            for (unsigned int x = 0; x < get_x_dim(); ++x) {
                                (*this)(w, z, y, x) = a;
                            }
                        }
                    }
                }
            }

            size_t bytes() const {
                return get_w_dim() * get_z_dim() *
                       get_y_dim() * get_x_dim() * sizeof(T);
            }

        private:
            size_t m_x_dim;
            size_t m_y_dim;
            size_t m_z_dim;
            size_t m_w_dim;
            bool   m_delete_buffer;
            T*           m_buffer;
    };

    
    /*               */
    typedef Array3D<std::complex<float>> Grid;

    
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

}


#endif
