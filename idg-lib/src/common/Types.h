#ifndef IDG_TYPES_H_
#define IDG_TYPES_H_

#include <iostream>
#include <ostream>
#include <complex>
#include <cassert>
#include <cstring>
#include <memory>
#include <vector>

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

} // end namespace idg

#endif
