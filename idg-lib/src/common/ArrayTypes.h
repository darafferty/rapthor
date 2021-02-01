// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_ARRAYTYPES_H
#define IDG_ARRAYTYPES_H

#include <complex>
#include <cassert>

#include <omp.h>

#include "auxiliary.h"
#include "Types.h"

namespace idg {

template <class T>
class Array1D {
 public:
  Array1D() : m_x_dim(0), m_memory(), m_buffer(nullptr) {}

  Array1D(size_t size)
      : m_x_dim(size),
        m_memory(new auxiliary::AlignedMemory(size * sizeof(T))),
        m_buffer((T*)m_memory->get()) {}

  Array1D(T* data, size_t size)
      : m_x_dim(size), m_memory(nullptr), m_buffer(data) {}

  Array1D(std::unique_ptr<auxiliary::Memory> memory, size_t size)
      : m_x_dim(size),
        m_memory(std::move(memory)),
        m_buffer((T*)m_memory->get()) {}

  Array1D(const Array1D& other) = delete;
  Array1D& operator=(const Array1D& rhs) = delete;

  Array1D(Array1D&& other)
      : m_x_dim(other.m_x_dim),
        m_memory(std::move(other.m_memory)),
        m_buffer(other.m_buffer) {
    other.m_buffer = nullptr;
  }

  Array1D& operator=(Array1D&& other) {
    m_x_dim = other.m_x_dim;
    m_memory = std::move(other.m_memory);
    m_buffer = other.m_buffer;
    other.m_buffer = nullptr;
    return *this;
  }

  virtual size_t size() const { return m_x_dim; }

  virtual size_t bytes() const { return size() * sizeof(T); }

  size_t get_x_dim() const { return m_x_dim; }

  T* data(size_t index = 0) const { return &m_buffer[index]; }

  bool contains_nan() const {
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

  bool contains_inf() const {
    volatile bool contains_inf = false;
#pragma omp parallel
    {
      // Normal openmp for loops do not allow break statements.
      // This custom loop will stop all threads at the first infinite value.
      const std::pair<size_t, size_t> range = omp_range();
      for (size_t i = range.first; i < range.second && !contains_inf; ++i) {
        if (!isfinite(*data(i))) {
          contains_inf = true;
        }
      }
    }
    return contains_inf;
  }

  void init(const T& a) {
#pragma omp parallel
    {
      // Use a single std::fill per thread instead of an openmp for loop.
      const std::pair<size_t, size_t> range = omp_range();
      std::fill(m_buffer + range.first, m_buffer + range.second, a);
    }
  }

  virtual void free() {
    m_x_dim = 0;
    m_memory.reset();
    m_buffer = nullptr;
  }

  void zero() {
    T zero;
    memset(static_cast<void*>(&zero), 0, sizeof(T));
    init(zero);
  }

  T& operator()(size_t i) { return m_buffer[i]; }

  const T& operator()(size_t i) const { return m_buffer[i]; }

 private:
  /**
   * @return 'start' and 'next' indices for use in openmp blocks.
   */
  std::pair<size_t, size_t> omp_range() const {
    const int thread_num = omp_get_thread_num();
    const bool last = (thread_num + 1 == omp_get_num_threads());
    const int chunk_size = size() / omp_get_num_threads();
    size_t start = chunk_size * thread_num;
    size_t next = last ? size() : start + chunk_size;
    return std::make_pair(start, next);
  }

 protected:
  size_t m_x_dim;
  std::unique_ptr<auxiliary::Memory> m_memory;
  T* m_buffer;
};

template <class T>
class Array2D : public Array1D<T> {
 public:
  Array2D(size_t y_dim = 0, size_t x_dim = 0)
      : Array1D<T>(y_dim * x_dim), m_y_dim(y_dim), m_x_dim(x_dim) {}

  Array2D(T* data, size_t y_dim, size_t x_dim)
      : Array1D<T>(data, y_dim * x_dim), m_y_dim(y_dim), m_x_dim(x_dim) {}

  Array2D(std::unique_ptr<auxiliary::Memory> memory, size_t y_dim, size_t x_dim)
      : Array1D<T>(std::move(memory), y_dim * x_dim),
        m_y_dim(y_dim),
        m_x_dim(x_dim) {}

  Array2D(Array2D&& other) {
    this->m_memory = std::move(other.m_memory);
    this->m_buffer = other.m_buffer;
    other.m_buffer = nullptr;
    m_y_dim = other.m_y_dim;
    m_x_dim = other.m_x_dim;
  }

  Array2D& operator=(Array2D&& other) {
    this->m_memory = std::move(other.m_memory);
    this->m_buffer = other.m_buffer;
    other.m_buffer = nullptr;
    m_y_dim = other.m_y_dim;
    m_x_dim = other.m_x_dim;
    return *this;
  }

  void free() override {
    Array1D<T>::free();
    m_x_dim = 0;
    m_y_dim = 0;
  }

  size_t get_x_dim() const { return m_x_dim; }
  size_t get_y_dim() const { return m_y_dim; }

  virtual size_t size() const override { return m_y_dim * m_x_dim; }

  inline size_t index(size_t y, size_t x) const { return y * m_x_dim + x; }

  T* data(size_t y = 0, size_t x = 0) const {
    return &this->m_buffer[index(y, x)];
  }

  const T& operator()(size_t y, size_t x) const {
    return this->m_buffer[index(y, x)];
  }

  T& operator()(size_t y, size_t x) { return this->m_buffer[index(y, x)]; }

 protected:
  size_t m_y_dim;
  size_t m_x_dim;
};

template <class T>
class Array3D : public Array1D<T> {
 public:
  Array3D(size_t z_dim = 0, size_t y_dim = 0, size_t x_dim = 0)
      : Array1D<T>(z_dim * y_dim * x_dim),
        m_z_dim(z_dim),
        m_y_dim(y_dim),
        m_x_dim(x_dim) {}

  Array3D(T* data, size_t z_dim, size_t y_dim, size_t x_dim)
      : Array1D<T>(data, z_dim * y_dim * x_dim),
        m_z_dim(z_dim),
        m_y_dim(y_dim),
        m_x_dim(x_dim) {}

  Array3D(std::unique_ptr<auxiliary::Memory> memory, size_t z_dim, size_t y_dim,
          size_t x_dim)
      : Array1D<T>(std::move(memory), z_dim * y_dim * x_dim),
        m_z_dim(z_dim),
        m_y_dim(y_dim),
        m_x_dim(x_dim) {}

  Array3D(Array3D&& other) {
    this->m_memory = std::move(other.m_memory);
    this->m_buffer = other.m_buffer;
    other.m_buffer = nullptr;
    m_z_dim = other.m_z_dim;
    m_y_dim = other.m_y_dim;
    m_x_dim = other.m_x_dim;
  }

  Array3D& operator=(Array3D&& other) {
    this->m_memory = std::move(other.m_memory);
    this->m_buffer = other.m_buffer;
    other.m_buffer = nullptr;
    m_z_dim = other.m_z_dim;
    m_y_dim = other.m_y_dim;
    m_x_dim = other.m_x_dim;
    return *this;
  }

  void free() override {
    Array1D<T>::free();
    m_x_dim = 0;
    m_y_dim = 0;
    m_z_dim = 0;
  }

  size_t get_x_dim() const { return m_x_dim; }
  size_t get_y_dim() const { return m_y_dim; }
  size_t get_z_dim() const { return m_z_dim; }

  virtual size_t size() const override { return m_z_dim * m_y_dim * m_x_dim; }

  inline size_t index(size_t z, size_t y, size_t x) const {
    return z * m_y_dim * m_x_dim + y * m_x_dim + x;
  }

  T* data(size_t z = 0, size_t y = 0, size_t x = 0) const {
    return &this->m_buffer[index(z, y, x)];
  }

  const T& operator()(size_t z, size_t y, size_t x) const {
    return this->m_buffer[index(z, y, x)];
  }

  T& operator()(size_t z, size_t y, size_t x) {
    return this->m_buffer[index(z, y, x)];
  }

 protected:
  size_t m_z_dim;
  size_t m_y_dim;
  size_t m_x_dim;
};

template <class T>
class Array4D : public Array1D<T> {
 public:
  Array4D(size_t w_dim = 0, size_t z_dim = 0, size_t y_dim = 0,
          size_t x_dim = 0)
      : Array1D<T>(w_dim * z_dim * y_dim * x_dim),
        m_w_dim(w_dim),
        m_z_dim(z_dim),
        m_y_dim(y_dim),
        m_x_dim(x_dim) {}

  Array4D(T* data, size_t w_dim, size_t z_dim, size_t y_dim, size_t x_dim)
      : Array1D<T>(data, w_dim * z_dim * y_dim * x_dim),
        m_w_dim(w_dim),
        m_z_dim(z_dim),
        m_y_dim(y_dim),
        m_x_dim(x_dim) {}

  Array4D(std::unique_ptr<auxiliary::Memory> memory, size_t w_dim, size_t z_dim,
          size_t y_dim, size_t x_dim)
      : Array1D<T>(std::move(memory), w_dim * z_dim * y_dim * x_dim),
        m_w_dim(w_dim),
        m_z_dim(z_dim),
        m_y_dim(y_dim),
        m_x_dim(x_dim) {}

  Array4D(Array4D&& other) {
    this->m_memory = std::move(other.m_memory);
    this->m_buffer = other.m_buffer;
    other.m_buffer = nullptr;
    m_w_dim = other.m_w_dim;
    m_z_dim = other.m_z_dim;
    m_y_dim = other.m_y_dim;
    m_x_dim = other.m_x_dim;
  }

  Array4D& operator=(Array4D&& other) {
    this->m_memory = std::move(other.m_memory);
    this->m_buffer = other.m_buffer;
    other.m_buffer = nullptr;
    m_w_dim = other.m_w_dim;
    m_z_dim = other.m_z_dim;
    m_y_dim = other.m_y_dim;
    m_x_dim = other.m_x_dim;
    return *this;
  }

  void free() override {
    Array1D<T>::free();
    m_x_dim = 0;
    m_y_dim = 0;
    m_z_dim = 0;
    m_w_dim = 0;
  }

  size_t get_x_dim() const { return m_x_dim; }
  size_t get_y_dim() const { return m_y_dim; }
  size_t get_z_dim() const { return m_z_dim; }
  size_t get_w_dim() const { return m_w_dim; }

  virtual size_t size() const override {
    return m_w_dim * m_z_dim * m_y_dim * m_x_dim;
  }

  inline size_t index(size_t w, size_t z, size_t y, size_t x) const {
    return w * m_z_dim * m_y_dim * m_x_dim + z * m_y_dim * m_x_dim +
           y * m_x_dim + x;
  }

  T* data(size_t w = 0, size_t z = 0, size_t y = 0, size_t x = 0) const {
    return &this->m_buffer[index(w, z, y, x)];
  }

  const T& operator()(size_t w, size_t z, size_t y, size_t x) const {
    return this->m_buffer[index(w, z, y, x)];
  }

  T& operator()(size_t w, size_t z, size_t y, size_t x) {
    return this->m_buffer[index(w, z, y, x)];
  }

 protected:
  size_t m_w_dim;
  size_t m_z_dim;
  size_t m_y_dim;
  size_t m_x_dim;
};

template <class T>
class Array5D : public Array1D<T> {
 public:
  Array5D(size_t e_dim, size_t d_dim = 0, size_t c_dim = 0, size_t b_dim = 0,
          size_t a_dim = 0)
      : Array1D<T>(e_dim * d_dim * c_dim * b_dim * a_dim),
        m_e_dim(e_dim),
        m_d_dim(d_dim),
        m_c_dim(c_dim),
        m_b_dim(b_dim),
        m_a_dim(a_dim) {}

  Array5D(T* data, size_t e_dim, size_t d_dim, size_t c_dim, size_t b_dim,
          size_t a_dim)
      : Array1D<T>(data, e_dim * d_dim * c_dim * b_dim * a_dim),
        m_e_dim(e_dim),
        m_d_dim(d_dim),
        m_c_dim(c_dim),
        m_b_dim(b_dim),
        m_a_dim(a_dim) {}

  Array5D(std::unique_ptr<auxiliary::Memory> memory, size_t e_dim, size_t d_dim,
          size_t z_dim, size_t c_dim, size_t b_dim, size_t a_dim)
      : Array1D<T>(std::move(memory), e_dim * d_dim * c_dim * b_dim * a_dim),
        m_e_dim(e_dim),
        m_d_dim(d_dim),
        m_c_dim(c_dim),
        m_b_dim(b_dim),
        m_a_dim(a_dim) {}

  Array5D(Array5D&& other) {
    this->m_memory = std::move(other.m_memory);
    this->m_buffer = other.m_buffer;
    other.m_buffer = nullptr;
    m_e_dim = other.m_e_dim;
    m_d_dim = other.m_d_dim;
    m_c_dim = other.m_c_dim;
    m_b_dim = other.m_b_dim;
    m_a_dim = other.m_a_dim;
  }

  Array5D& operator=(Array5D&& other) {
    this->m_memory = std::move(other.m_memory);
    this->m_buffer = other.m_buffer;
    other.m_buffer = nullptr;
    m_e_dim = other.m_e_dim;
    m_d_dim = other.m_d_dim;
    m_c_dim = other.m_c_dim;
    m_b_dim = other.m_b_dim;
    m_a_dim = other.m_a_dim;
    return *this;
  }

  void free() override {
    Array1D<T>::free();
    m_a_dim = 0;
    m_b_dim = 0;
    m_c_dim = 0;
    m_d_dim = 0;
    m_e_dim = 0;
  }

  size_t get_a_dim() const { return m_a_dim; }
  size_t get_b_dim() const { return m_b_dim; }
  size_t get_c_dim() const { return m_c_dim; }
  size_t get_d_dim() const { return m_d_dim; }
  size_t get_e_dim() const { return m_e_dim; }

  virtual size_t size() const override {
    return m_e_dim * m_d_dim * m_c_dim * m_b_dim * m_a_dim;
  }

  inline size_t index(size_t e, size_t d, size_t c, size_t b, size_t a) const {
    return e * m_d_dim * m_c_dim * m_b_dim * m_a_dim +
           d * m_c_dim * m_b_dim * m_a_dim + c * m_b_dim * m_a_dim +
           b * m_a_dim + a;
  }

  T* data(size_t e = 0, size_t d = 0, size_t c = 0, size_t b = 0,
          size_t a = 0) const {
    return &this->m_buffer[index(e, d, c, b, a)];
  }

  const T& operator()(size_t e = 0, size_t d = 0, size_t c = 0, size_t b = 0,
                      size_t a = 0) const {
    return this->m_buffer[index(e, d, c, b, a)];
  }

  T& operator()(size_t e = 0, size_t d = 0, size_t c = 0, size_t b = 0,
                size_t a = 0) {
    return this->m_buffer[index(e, d, c, b, a)];
  }

 protected:
  size_t m_e_dim;
  size_t m_d_dim;
  size_t m_c_dim;
  size_t m_b_dim;
  size_t m_a_dim;
};

using Grid = Array4D<std::complex<float>>;

template <class T>
std::ostream& operator<<(std::ostream& os, const Array1D<T>& a) {
  for (unsigned int x = 0; x < a.get_x_dim(); ++x) {
    os << a(x);
    if (x != a.get_x_dim() - 1) {
      os << ",";
    }
  }
  os << std::endl;
  return os;
}

template <class T>
std::ostream& operator<<(std::ostream& os, const Array2D<T>& a) {
  for (unsigned int y = 0; y < a.get_y_dim(); ++y) {
    for (unsigned int x = 0; x < a.get_x_dim(); ++x) {
      os << a(y, x);
      if (x != a.get_x_dim() - 1) {
        os << ",";
      }
    }
    os << std::endl;
  }
  return os;
}

template <class T>
std::ostream& operator<<(std::ostream& os, const Array3D<T>& a) {
  for (unsigned int z = 0; z < a.get_z_dim(); ++z) {
    os << std::endl;
    for (unsigned int y = 0; y < a.get_y_dim(); ++y) {
      for (unsigned int x = 0; x < a.get_x_dim(); ++x) {
        os << a(z, y, x);
        if (x != a.get_x_dim() - 1) {
          os << ",";
        }
      }
      os << std::endl;
    }
  }
  return os;
}

template <class T>
std::ostream& operator<<(std::ostream& os, const Array4D<T>& a) {
  for (unsigned int w = 0; w < a.get_w_dim(); ++w) {
    os << std::endl;
    for (unsigned int z = 0; z < a.get_z_dim(); ++z) {
      os << std::endl;
      for (unsigned int y = 0; y < a.get_y_dim(); ++y) {
        for (unsigned int x = 0; x < a.get_x_dim(); ++x) {
          os << a(w, z, y, x);
          if (x != a.get_x_dim() - 1) {
            os << ",";
          }
        }
        os << std::endl;
      }
    }
  }
  return os;
}

}  // end namespace idg

#endif
