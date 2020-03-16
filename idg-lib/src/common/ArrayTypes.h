#ifndef IDG_ARRAYTYPES_H
#define IDG_ARRAYTYPES_H

#include <complex>
#include <cassert>

#include <omp.h>

#include "auxiliary.h"
#include "Types.h"

namespace idg {

template<class T>
class ArrayXD {
    public:
        ArrayXD() :
            m_shape(0),
            m_memory(0),
            m_buffer(nullptr)
        {
        }

        ArrayXD(
            std::vector<size_t> shape) :
            m_shape(shape),
            m_memory(std::shared_ptr<auxiliary::Memory>(new auxiliary::AlignedMemory(size() * sizeof(T)))),
            m_buffer((T*) m_memory->get())
        {
        }

        ArrayXD(
            T* data,
            std::vector<size_t> shape) :
            m_shape(shape),
            m_memory(nullptr),
            m_buffer(data)
        {
        }

        ArrayXD(
            std::shared_ptr<auxiliary::Memory> memory,
            std::vector<size_t> shape) :
            m_shape(shape),
            m_memory(memory),
            m_buffer((T*) m_memory->get())
        {
        }

        ArrayXD(const ArrayXD& other) = delete;
        ArrayXD& operator=(const ArrayXD& rhs) = delete;

        ArrayXD(ArrayXD&& other) :
            m_shape(other.m_shape),
            m_memory(other.m_memory),
            m_buffer(other.m_buffer)
        {
            other.m_buffer = nullptr;
        }

        // move assignment operator
        ArrayXD& operator=(ArrayXD&& other)
        {
            m_shape = other.m_shape;
            m_buffer = other.m_buffer;
            m_memory =  other.m_memory;
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
            return &m_buffer[index];
        }

        bool contains_nan() const
        {
            volatile bool contains_nan = false;
            #pragma omp parallel for
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
            #pragma omp parallel for
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
            #pragma omp parallel for
            for (size_t i = 0; i < size(); ++i) {
                m_buffer[i] = a;
            }
        }

        void zero()
        {
            T zero;
            memset((void *) &zero, 0, sizeof(T));
            init(zero);
        }

        T& operator()(
            size_t i)
        {
            return m_buffer[i];
        }

        const T& operator()(
            size_t i) const
        {
            return m_buffer[i];
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

    protected:
        std::vector<size_t> m_shape;
        std::shared_ptr<auxiliary::Memory> m_memory;
        T* m_buffer;
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

        Array1D(
            std::shared_ptr<auxiliary::Memory> memory,
            size_t width) :
            ArrayXD<T>(memory, {width})
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

        Array2D(
            std::shared_ptr<auxiliary::Memory> memory,
            size_t y_dim,
            size_t x_dim) :
            ArrayXD<T>(memory, {y_dim, x_dim})
        {}

        size_t get_x_dim() const { return this->get_n_dim(1); }
        size_t get_y_dim() const { return this->get_n_dim(0); }

        inline size_t index(
            size_t y,
            size_t x) const
        {
            return y * this->m_shape[1] + x;
        }

        T* data(
            size_t y = 0,
            size_t x = 0) const
        {
            return &this->m_buffer[this->index(y, x)];
        }

        const T& operator()(
            size_t y,
            size_t x) const
        {
            return this->m_buffer[this->index(y, x)];
        }

        T& operator()(
            size_t y,
            size_t x)
        {
            return this->m_buffer[this->index(y, x)];
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

        Array3D(
            std::shared_ptr<auxiliary::Memory> memory,
            size_t z_dim,
            size_t y_dim,
            size_t x_dim) :
            ArrayXD<T>(memory, {z_dim, y_dim, x_dim})
        {}

        size_t get_x_dim() const { return this->get_n_dim(2); }
        size_t get_y_dim() const { return this->get_n_dim(1); }
        size_t get_z_dim() const { return this->get_n_dim(0); }

        inline size_t index(
            size_t z,
            size_t y,
            size_t x) const
        {
            return z * this->m_shape[1] * this->m_shape[2] +
                                      y * this->m_shape[2] +
                                                         x;
        }

        T* data(
            size_t z = 0,
            size_t y = 0,
            size_t x = 0) const
        {
            return &this->m_buffer[index(z, y, x)];
        }

        const T& operator()(
            size_t z,
            size_t y,
            size_t x) const
        {
            return this->m_buffer[this->index(z, y, x)];
        }

        T& operator()(
            size_t z,
            size_t y,
            size_t x)
        {
            return this->m_buffer[this->index(z, y, x)];
        }
};


template<class T>
class Array4D : public ArrayXD<T> {
    using ArrayXD<T>::ArrayXD;

    public:
        Array4D() {};

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

        Array4D(
            std::shared_ptr<auxiliary::Memory> memory,
            size_t w_dim,
            size_t z_dim,
            size_t y_dim,
            size_t x_dim) :
            ArrayXD<T>(memory, {w_dim, z_dim, y_dim, x_dim})
        {}

        size_t get_x_dim() const { return this->get_n_dim(3); }
        size_t get_y_dim() const { return this->get_n_dim(2); }
        size_t get_z_dim() const { return this->get_n_dim(1); }
        size_t get_w_dim() const { return this->get_n_dim(0); }

        inline size_t index(
            size_t w,
            size_t z,
            size_t y,
            size_t x) const
        {
            return w * this->m_shape[1] * this->m_shape[2] * this->m_shape[3] +
                                      z * this->m_shape[2] * this->m_shape[3] +
                                                         y * this->m_shape[3] +
                                                                            x;
        }

        T* data(
            size_t w = 0,
            size_t z = 0,
            size_t y = 0,
            size_t x = 0) const
        {
            return &this->m_buffer[index(w, z, y, x)];
        }

        const T& operator()(
            size_t w,
            size_t z,
            size_t y,
            size_t x) const
        {
            return this->m_buffer[index(w, z, y, x)];
        }

        T& operator()(
            size_t w,
            size_t z,
            size_t y,
            size_t x)
        {
            return this->m_buffer[index(w, z, y, x)];
        }
};

using Grid = Array4D<std::complex<float>>;

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