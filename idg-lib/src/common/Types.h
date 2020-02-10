#ifndef IDG_TYPES_H_
#define IDG_TYPES_H_

#include <iostream>
#include <ostream>
#include <complex>
#include <cassert>
#include <cstring>
#include <memory>
#include <vector>

#define ALIGNMENT       64
#define WTILE_SIZE      128

#ifndef FUNCTION_ATTRIBUTES
#define FUNCTION_ATTRIBUTES
#endif

namespace idg {

    /* Structures */
    typedef struct { float real; float imag; } float2;
    typedef struct { double real; double imag; } double2;
    #include "KernelTypes.h"

    template<class T>
    struct Matrix2x2 {T xx; T xy; T yx; T yy;};

    template<class T>
    using Visibility = Matrix2x2<T>;

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


    template<class T>
    T* allocate_memory(size_t n) {
        void *ptr = nullptr;
        if (n > 0) {
            size_t bytes = n * sizeof(T);
            bytes = (((bytes - 1) / ALIGNMENT) * ALIGNMENT) + ALIGNMENT;
            if (posix_memalign(&ptr, ALIGNMENT, bytes) != 0) {
                std::cerr << "Could not allocate " << bytes << " bytes" << std::endl;
                exit(EXIT_FAILURE);
            };
        }
        return (T *) ptr;
    }


    /* Debugging */
    template<typename T>
    inline bool isnan(T& value) {
        return (std::isnan(value));
    }

    template<typename T>
    inline bool isnan(std::complex<T>& value) {
        return (std::isnan(value.real()) || std::isnan(value.imag()));
    }

    template<typename T>
    inline bool isnan(Matrix2x2<std::complex<T>>& m) {
        return (isnan(m.xx) || isnan(m.xy) || isnan(m.yx) || isnan(m.yy));
    }

    template<typename T>
    inline bool isnan(UVW<T>& uvw) {
        return (std::isnan(uvw.u) || std::isnan(uvw.v) || std::isnan(uvw.w));
    }

    template<typename T>
    inline bool isfinite(T& value) {
        return (std::isfinite(value));
    }

    template<typename T>
    inline bool isfinite(std::complex<T>& value) {
        return (std::isfinite(value.real()) && std::isfinite(value.imag()));
    }

    template<typename T>
    inline bool isfinite(Matrix2x2<std::complex<T>>& m) {
        return (isfinite(m.xx) && isfinite(m.xy) && isfinite(m.yx) && isfinite(m.yy));
    }

    template<typename T>
    inline bool isfinite(UVW<T>& uvw) {
        return (std::isfinite(uvw.u) && std::isfinite(uvw.v) && std::isfinite(uvw.w));
    }

    /* Classes */
    template<class T>
    class ArrayXD {
        public:
            ArrayXD() :
                m_shape(0),
                m_buffer(nullptr)
            {
            }

            ArrayXD(
                std::vector<size_t> shape) :
                m_shape(shape),
                m_buffer(allocate_memory<T>(size()), &free) // shared_ptr with custom deleter that deletes an array
            {
            }

            ArrayXD(
                T* data,
                std::vector<size_t> shape) :
                m_shape(shape),
                m_buffer(data, [](T*){}) // shared_ptr with custom deleter that does nothing
            {
            }

            ArrayXD(const ArrayXD& other) = delete;
            ArrayXD& operator=(const ArrayXD& rhs) = delete;

            ArrayXD(ArrayXD&& other) :
                m_shape(other.m_shape),
                m_buffer(other.m_buffer)
            {
                other.m_buffer = nullptr;
            }

            // move assignment operator
            ArrayXD& operator=(ArrayXD&& other)
            {
                m_shape = other.m_shape;
                m_buffer = other.m_buffer;
                other.m_buffer = nullptr;
				return *this;
            }

            ~ArrayXD() {}

            std::vector<size_t> shape() const
            {
                return m_shape;
            }

            size_t size() const
            {
                size_t result = 1;
                for (auto n : m_shape) {
                    result *= n;
                }
                return result;
            }

            size_t bytes() const
            {
                return size() * sizeof(T);
            }

            T* data(
                size_t index = 0) const
            {
                return &m_buffer.get()[index];
            }

            bool contains_nan() const
            {
                volatile bool contains_nan = false;
                for (size_t i = 0; i < size(); i++) {
                    if (contains_nan) {
                        continue;
                    }
                    T* value = data(i);
                    if (isnan(*value)) {
                        contains_nan = true;
                    }
                }
                return contains_nan;
            };

            bool contains_inf() const
            {
                volatile bool contains_inf = false;
                for (size_t i = 0; i < size(); i++) {
                    if (contains_inf) {
                        continue;
                    }
                    T* value = data(i);
                    if (!isfinite(*value)) {
                        contains_inf = true;
                    }
                }
                return contains_inf;
            }

            void init(const T& a)
            {
                for (size_t i = 0; i < size(); ++i) {
                    m_buffer.get()[i] = a;
                }
            }

            void zero()
            {
                memset((void *) m_buffer.get(), 0, bytes());
            }

            T& operator()(
                size_t i)
            {
                return m_buffer.get()[i];
            }

            const T& operator()(
                size_t i) const
            {
                return m_buffer.get()[i];
            }

            // TODO: if the buffer is not owned, there is no guarantee that it won't be destroyed.
            // Need to return a copy in that case.
            const std::shared_ptr<const T> get() const
            {
                return m_buffer;
            }

        protected:
            size_t get_n_dim(
                size_t n) const
            {
                assert(n < m_shape.size());
                return m_shape[n];
            }

            size_t index(
                std::vector<size_t> idx) const
            {
                assert(idx.size() == m_shape.size());
                size_t result = 0;
                for (unsigned i = 0; i < idx.size(); i++) {
                    size_t temp = idx[i];
                    for (unsigned j = i + 1; j < idx.size(); j++) {
                        temp *= m_shape[j];
                    }
                    result += temp;
                }
                return result;
            }

        protected:
            std::vector<size_t> m_shape;
            std::shared_ptr<T> m_buffer;
            bool m_delete_buffer;
    };

    template<class T>
    class Array1D : public ArrayXD<T> {
        using ArrayXD<T>::ArrayXD;

        public:
            Array1D(
                size_t width = 0) :
                ArrayXD<T>({width})
            {}

            Array1D(
                T* data,
                size_t width) :
                ArrayXD<T>(data, {width})
            {}

            size_t get_x_dim() const { return this->get_n_dim(0); }
    };

    template<class T>
    class Array2D : public ArrayXD<T> {
        using ArrayXD<T>::ArrayXD;

        public:
            Array2D(
                size_t y_dim = 0,
                size_t x_dim = 0) :
                ArrayXD<T>({y_dim, x_dim})
            {}

            Array2D(
                T* data,
                size_t y_dim,
                size_t x_dim) :
                ArrayXD<T>(data, {y_dim, x_dim})
            {}

            size_t get_x_dim() const { return this->get_n_dim(1); }
            size_t get_y_dim() const { return this->get_n_dim(0); }

            T* data(
                size_t y = 0,
                size_t x = 0) const
            {
                return &this->m_buffer.get()[this->index({y, x})];
            }

            const T& operator()(
                size_t y,
                size_t x) const
            {
                return this->m_buffer.get()[this->index({y, x})];
            }

            T& operator()(
                size_t y,
                size_t x)
            {
                return this->m_buffer.get()[this->index({y, x})];
            }
    };


    template<class T>
    class Array3D : public ArrayXD<T> {
        using ArrayXD<T>::ArrayXD;

        public:
            Array3D(
                size_t z_dim = 0,
                size_t y_dim = 0,
                size_t x_dim = 0) :
                ArrayXD<T>({z_dim, y_dim, x_dim})
            {}

            Array3D(
                T* data,
                size_t z_dim,
                size_t y_dim,
                size_t x_dim) :
                ArrayXD<T>(data, {z_dim, y_dim, x_dim})
            {}

            size_t get_x_dim() const { return this->get_n_dim(2); }
            size_t get_y_dim() const { return this->get_n_dim(1); }
            size_t get_z_dim() const { return this->get_n_dim(0); }

            T* data(
                size_t z = 0,
                size_t y = 0,
                size_t x = 0) const
            {
                return &this->m_buffer.get()[this->index({z, y, x})];
            }

            const T& operator()(
                size_t z,
                size_t y,
                size_t x) const
            {
                return this->m_buffer.get()[this->index({z, y, x})];
            }

            T& operator()(
                size_t z,
                size_t y,
                size_t x)
            {
                return this->m_buffer.get()[this->index({z, y, x})];
            }
    };


    template<class T>
    class Array4D : public ArrayXD<T> {
        using ArrayXD<T>::ArrayXD;

        public:
            Array4D(
                size_t w_dim,
                size_t z_dim,
                size_t y_dim,
                size_t x_dim) :
                ArrayXD<T>({w_dim, z_dim, y_dim, x_dim})
            {}

            Array4D(
                T* data,
                size_t w_dim,
                size_t z_dim,
                size_t y_dim,
                size_t x_dim) :
                ArrayXD<T>(data, {w_dim, z_dim, y_dim, x_dim})
            {}

            size_t get_x_dim() const { return this->get_n_dim(3); }
            size_t get_y_dim() const { return this->get_n_dim(2); }
            size_t get_z_dim() const { return this->get_n_dim(1); }
            size_t get_w_dim() const { return this->get_n_dim(0); }

            T* data(
                size_t w = 0,
                size_t z = 0,
                size_t y = 0,
                size_t x = 0) const
            {
                return &this->m_buffer.get()[this->index({w, z, y, x})];
            }

            const T& operator()(
                size_t w,
                size_t z,
                size_t y,
                size_t x) const
            {
                return this->m_buffer.get()[this->index({w, z, y, x})];
            }

            T& operator()(
                size_t w,
                size_t z,
                size_t y,
                size_t x)
            {
                return this->m_buffer.get()[this->index({w, z, y, x})];
            }
    };

    using Grid = Array4D<std::complex<float>>;

    /* Output */
    std::ostream& operator<<(std::ostream& os, Baseline& b);
    std::ostream& operator<<(std::ostream& os, Coordinate& c);
    std::ostream& operator<<(std::ostream& os, Metadata& m);

    template<class T>
    std::ostream& operator<<(std::ostream &out, Matrix2x2<std::complex<T>>& m) {
        out << "("
            << m.xx << ","
            << m.xy << ","
            << m.yx << ","
            << m.yy << ")";
        return out;
    }

    template<class T>
    std::ostream& operator<<(std::ostream& os, UVW<T>& uvw);

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

#endif
