/*
 * Datatypes.h
 *
 */

#ifndef IDG_DATATYPES_H_
#define IDG_DATATYPES_H_

#include <iostream>
#include <vector>
#include <utility>

namespace idg {

    using TimeIndex = int;
    using AtermIndex = int;

    template<class T>
    struct UVWCoordinate {T u; T v; T w;};

    template<class T>
    using UVWCoordinates = std::vector<UVWCoordinate<T>>;

    template<class T>
    using Frequencies = std::vector<T>;

    template<class T>
    using Wavenumbers = std::vector<T>;

    template<class T>
    struct StationPair {T first; T second;};

    template<class T>
    using StationPairs = std::vector<StationPair<T>>;

    /* using float2 = struct {float real; float imag; }; */
    /* using double2 = struct {double real; double imag; }; */

    template<class T>
    struct Matrix2x2 {T xx; T xy; T yx; T yy;};

    template<class T>
    using Aterm = Matrix2x2<T>;

    template<class T>
    using Visibility = Matrix2x2<T>;

    template<class T>
    using Visibilities = std::vector<Visibility<T>>;

    template<class T>
    using VisibilityGroup = std::vector<Visibility<T>>;

    template<class T>
    using VisibilityGroups = std::vector<VisibilityGroup<T>>;


    template<class T>
    class Grid2D {
    public:
        Grid2D(size_t height=1,
               size_t width=1)
            :  m_width(width),
               m_height(height),
               m_delete_buffer(true),
               m_buffer(new T[height*width])
        {}

        Grid2D(T* data, size_t height=1, size_t width=1)
            :  m_width(width),
               m_height(height),
               m_delete_buffer(false),
               m_buffer(data)
        {}

        Grid2D(const Grid2D& v) = delete;
        Grid2D& operator=(const Grid2D& rhs) = delete;
        virtual ~Grid2D() { if (m_delete_buffer) delete[] m_buffer; }

        T* data(size_t row=0) const { return &m_buffer[row*m_width]; }

        size_t get_width() const { return m_width; }
        size_t get_height() const { return m_height; }

        void reserve(size_t height=1, size_t width=1)
        {
            m_width = width;
            m_height = height;
            delete[] m_buffer; // Note data is lost
            m_buffer = new T[height*width];
        }

        const T& operator()(size_t y,
                            size_t x) const
        {
            return m_buffer[x + m_width*y];
        }

        T& operator()(size_t y,
                      size_t x)
        {
            return m_buffer[x + m_width*y];
        }

        void init(const T& a) {
            for (unsigned int y = 0; y < get_height(); ++y)
                for (unsigned int x = 0; x < get_width(); ++x)
                    (*this)(y, x) = a;
        }

    private:
        size_t     m_width;
        size_t     m_height;
        const bool m_delete_buffer;
        T*         m_buffer;
    };


    template<class T>
    class Grid3D {
    public:
        Grid3D(size_t depth=1, size_t height=1,
               size_t width=1)
            :  m_width(width),
               m_height(height),
               m_depth(depth),
               m_delete_buffer(true),
               m_buffer(new T[height*width*depth])
        {}

        Grid3D(T* data, size_t depth=1,
               size_t height=1, size_t width=1)
            :  m_width(width),
               m_height(height),
               m_depth(depth),
               m_delete_buffer(false),
               m_buffer(data)
        {}

        Grid3D(const Grid3D& v) = delete;
        Grid3D& operator=(const Grid3D& rhs) = delete;
        virtual ~Grid3D() { if (m_delete_buffer) delete[] m_buffer; }

        T* data(size_t layer=0) const { return &m_buffer[layer*m_height*m_width]; }

        size_t get_width() const { return m_width; }
        size_t get_height() const { return m_height; }
        size_t get_depth() const { return m_depth; }

        void reserve(size_t depth=1, size_t height=1, size_t width=1)
        {
            m_width = width;
            m_height = height;
            m_depth = depth;
            delete[] m_buffer; // Note data is lost
            m_buffer = new T[height*width*depth];
        }

        const T& operator()(size_t layer,
                            size_t y,
                            size_t x) const
        {
            return m_buffer[x + m_width*y + m_width*m_height*layer];
        }

        T& operator()(size_t layer,
                      size_t y,
                      size_t x)
        {
            return m_buffer[x + m_width*y + m_width*m_height*layer];
        }

        void init(const T& a) {
            for (unsigned int z = 0; z < get_depth(); ++z)
                for (unsigned int y = 0; y < get_height(); ++y)
                    for (unsigned int x = 0; x < get_width(); ++x)
                        (*this)(z, y, x) = a;
        }

    private:
        size_t m_width;
        size_t m_height;
        size_t m_depth;
        const bool m_delete_buffer;
        T* m_buffer;
    };


    // auxiliary methods

    template<class T>
    std::ostream& operator<<(std::ostream& os,
                             const UVWCoordinate<T>& c)
    {
        os << "{" << c.u << "," << c.v
           << "," << c.w << "}";
        return os;
    }


    template<class T>
    std::ostream& operator<<(std::ostream& os,
                             const StationPair<T>& s)
    {
        os << "{" << s.first << "," << s.second << "}";
        return os;
    }


    /* std::ostream& operator<<(std::ostream& os, */
    /*                          const float2& x) */
    /* { */
    /*     os << "(" << x.real << "," << x.imag << ")"; */
    /*     return os; */
    /* } */


    /* std::ostream& operator<<(std::ostream& os, */
    /*                          const double2& x) */
    /* { */
    /*     os << "(" << x.real << "," << x.imag << ")"; */
    /*     return os; */
    /* } */


    template<class T>
    std::ostream& operator<<(std::ostream& os,
                             const Matrix2x2<T>& A)
    {
        os << "[" << A.xx << "," << A.xy << ";" << std::endl
            << " " << A.yx << "," << A.yy << "]";
        return os;
    }


    template<class T>
    std::ostream& operator<<(std::ostream& os,
                             const Grid2D<T>& grid)
    {
        for (unsigned int y = 0; y < grid.get_height(); ++y) {
            for (unsigned int x = 0; x < grid.get_width(); ++x) {
                os << grid(y,x);
                if (x != grid.get_width()-1) {
                    os << ",";
                }
            }
            os << std::endl;
        }
        return os;
    }


    template<class T>
    std::ostream& operator<<(std::ostream& os,
                             const Grid3D<T>& grid)
    {
        for (unsigned int z = 0; z < grid.get_depth(); ++z) {
            os << "layer==" << z << ":" << std::endl;
            for (unsigned int y = 0; y < grid.get_height(); ++y) {
                for (unsigned int x = 0; x < grid.get_width(); ++x) {
                    os << grid(z,y,x);
                    if (x != grid.get_width()-1) {
                        os << ",";
                    }
                }
                os << std::endl;
            }
        }
        return os;
    }


    /* float2 operator*(const float2& x, */
    /*                  const float2& y) */
    /* { */
    /*     return {x.real*y.real - x.imag*y.imag, */
    /*             x.real*y.imag + x.imag*y.real}; */
    /* } */

    /* float2 operator+(const float2& x, */
    /*                  const float2& y) */
    /* { */
    /*     return {x.real + y.real, x.imag + y.imag}; */
    /* } */

    /* double2 operator*(const double2& x, */
    /*                   const double2& y) */
    /* { */
    /*     return {x.real*y.real - x.imag*y.imag, */
    /*             x.real*y.imag + x.imag*y.real}; */
    /* } */

    /* double2 operator+(const double2& x, */
    /*                   const double2& y) */
    /* { */
    /*     return {x.real + y.real, x.imag + y.imag}; */
    /* } */


    template<class T>
    Matrix2x2<T> operator*(const Matrix2x2<T>& A,
                           const Matrix2x2<T>& B)
    {
        return {A.xx*B.xx + A.xy*B.yx,
                A.xx*B.xy + A.xy*B.yy,
                A.yx*B.xx + A.yy*B.yx,
                A.yx*B.xy + A.yy*B.yy};
    }

    template<class T>
    Matrix2x2<T> operator+(const Matrix2x2<T>& A,
                           const Matrix2x2<T>& B)
    {
        return {A.xx + B.xx, A.xy + B.xy,
                A.yx + B.yx, A.yy + B.yy};
    }

}

#endif /* DATATYPES_H_ */
