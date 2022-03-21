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
  /**
   * @brief Construct an empty Array1D object
   *
   */
  Array1D() : m_x_dim(0), m_memory(), m_buffer(nullptr) {}

  /**
   * @brief Construct a new Array1D object
   *
   * The memory is allocated through an auxiliary::AlignedMemory object
   *
   * @param size Size of the new array in number of elements
   */
  Array1D(size_t size)
      : m_x_dim(size),
        m_memory(new auxiliary::AlignedMemory(size * sizeof(T))),
        m_buffer((T*)m_memory->data()) {}

  /**
   * @brief Construct a new Array1D object from a raw pointer
   *
   * @param data Pointer to the memory that the Array will use.
   *             Must be avaible during the life time of the Array
   *             The user is responsible for allocation and deallocation.
   *
   * @param size Size of the Array in number of elements
   */
  Array1D(T* data, size_t size)
      : m_x_dim(size), m_memory(nullptr), m_buffer(data) {}

  /**
   * @brief Construct a new Array 1 D object from a Memory object
   *
   * @param memory Unique pointer to the Memory object that will hold the data.
   *               Must be large enough to hold all elements.
   *               The Array class will take ownership of the unique_ptr by
   * moving it.
   * @param size
   */
  Array1D(std::unique_ptr<auxiliary::Memory> memory, size_t size)
      : m_x_dim(size),
        m_memory(std::move(memory)),
        m_buffer((T*)m_memory->data()) {}

  Array1D(const Array1D& other) = delete;
  Array1D& operator=(const Array1D& rhs) = delete;

  /**
   * @brief Move constructor
   *
   * If the memory was managed by the other Array, the new
   * Array will take over ownership.
   *
   * @param other Pointer to the other Array
   */
  Array1D(Array1D&& other)
      : m_x_dim(other.m_x_dim),
        m_memory(std::move(other.m_memory)),
        m_buffer(other.m_buffer) {
    other.m_buffer = nullptr;
  }

  /**
   * @brief Move assignment operator
   *
   * If the memory was managed by the other Array, the new
   * Array will take over ownership.
   *
   * @param other Pointer to the other Array
   * @return Reference to the newly constructed Array\\
   */
  Array1D& operator=(Array1D&& other) {
    m_x_dim = other.m_x_dim;
    m_memory = std::move(other.m_memory);
    m_buffer = other.m_buffer;
    other.m_buffer = nullptr;
    return *this;
  }

  /**
   * @brief Get the size of the Array in number of elements
   *
   * @return Size of the Array in number of elements
   */
  virtual size_t size() const { return m_x_dim; }

  /**
   * @brief Get the size of the Array in number of bytes
   *
   * @return Size of the Array in number of bytes
   */
  virtual size_t bytes() const { return size() * sizeof(T); }

  /**
   * @brief Get the length of the x dimension
   *
   * @return Length of the x dimension
   */
  size_t get_x_dim() const { return m_x_dim; }

  /**
   * @brief Get a pointer to the data
   *
   * @param index Optional offset
   * @return Pointer to the data
   */
  T* data(size_t index = 0) const { return &m_buffer[index]; }

  /**
   * @brief Check whether the Array contains NaN values
   *
   * @return true if the Array contains at least one NaN value
   */
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

  /**
   * @brief Check whether the Array contains Inf values
   *
   * @return true if the Array contains at least one non finite value
   */
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

  /**
   * @brief Initialize all elements to some value
   *
   * @param a Value to initialize the elements to
   */
  void init(const T& a) {
#pragma omp parallel
    {
      // Use a single std::fill per thread instead of an openmp for loop.
      const std::pair<size_t, size_t> range = omp_range();
      std::fill(m_buffer + range.first, m_buffer + range.second, a);
    }
  }

  /**
   * @brief Free memory, reset Array to zero length
   *
   * Do not call on Arrays with unmanaged memory, created from raw pointers
   *
   */
  virtual void free() {
    m_x_dim = 0;
    m_memory.reset();
    m_buffer = nullptr;
  }

  /**
   * @brief Set all elements to zero
   *
   */
  void zero() {
    T zero;
    memset(static_cast<void*>(&zero), 0, sizeof(T));
    init(zero);
  }

  /**
   * @brief Indexing operator
   *
   * @param i Index of element
   * @return Reference to the value of the i-th element
   */
  T& operator()(size_t i) { return m_buffer[i]; }

  /**
   * @brief Indexing operator
   *
   * @param i Index of element
   * @return Const reference to the value of the i-th element
   */
  const T& operator()(size_t i) const { return m_buffer[i]; }

 private:
  /**
   * @return 'start' and 'next' indices for use in openmp blocks.
   */
  std::pair<size_t, size_t> omp_range() const {
    const size_t thread_num = omp_get_thread_num();
    const bool last = (thread_num + 1 == size_t(omp_get_num_threads()));
    const size_t chunk_size = size() / omp_get_num_threads();
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
  /**
   * @brief Construct a new Array2D object
   *
   * The memory is allocated through an auxiliary::AlignedMemory object
   *
   * @param y_dim Size of the y dimension
   * @param x_dim Size of the x dimension (fastest changing index)
   */
  Array2D(size_t y_dim = 0, size_t x_dim = 0)
      : Array1D<T>(y_dim * x_dim), m_y_dim(y_dim), m_x_dim(x_dim) {}

  /**
   * @brief Construct a new Array2D object from a raw pointer
   *
   * @param data Pointer to the memory that the Array will use.
   *             Must be avaible during the life time of the Array
   *             The user is responsible for allocation and deallocation.
   * @param y_dim Size of the y dimension
   * @param x_dim Size of the x dimension (fastest changing index)
   */
  Array2D(T* data, size_t y_dim, size_t x_dim)
      : Array1D<T>(data, y_dim * x_dim), m_y_dim(y_dim), m_x_dim(x_dim) {}

  /**
   * @brief Construct a new Array2D object from a Memory object
   *
   * @param memory Unique pointer to the Memory object that will hold the data.
   *               Must be large enough to hold all elements.
   *               The Array class will take ownership of the unique_ptr by
   * moving it.
   * @param y_dim Size of the y dimension
   * @param x_dim Size of the x dimension (fastest changing index)
   */
  Array2D(std::unique_ptr<auxiliary::Memory> memory, size_t y_dim, size_t x_dim)
      : Array1D<T>(std::move(memory), y_dim * x_dim),
        m_y_dim(y_dim),
        m_x_dim(x_dim) {}

  /**
   * @brief Move constructor
   *
   * If the memory was managed by the other Array, the new
   * Array will take over ownership.
   *
   * @param other Pointer to the other Array
   */
  Array2D(Array2D&& other) {
    this->m_memory = std::move(other.m_memory);
    this->m_buffer = other.m_buffer;
    other.m_buffer = nullptr;
    m_y_dim = other.m_y_dim;
    m_x_dim = other.m_x_dim;
  }

  /**
   * @brief Move assignment operator
   *
   * If the memory was managed by the other Array, the new
   * Array will take over ownership.
   *
   * @param other Pointer to the other Array
   * @return Reference to the newly constructed Array\\
   */
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

  /**
   * @brief Get the length of the x dimension
   *
   * @return Length of the x dimension
   */
  size_t get_x_dim() const { return m_x_dim; }

  /**
   * @brief Get the length of the y dimension
   *
   * @return Length of the y dimension
   */
  size_t get_y_dim() const { return m_y_dim; }

  size_t size() const override { return m_y_dim * m_x_dim; }

  /**
   * @brief Compute the linear index from multi dimensional indices
   *
   * @param y Index along the y-axis
   * @param x Index along the x-axis (fastest changing index)
   * @return linear index of element at y,x
   */
  inline size_t index(size_t y, size_t x) const { return y * m_x_dim + x; }

  /**
   * @brief Get a pointer to the data at position y, x
   *
   * @param y Index along the y-axis
   * @param x Index along the x-axis (fastest changing index)
   * @return T*
   */
  T* data(size_t y = 0, size_t x = 0) const {
    return &this->m_buffer[index(y, x)];
  }

  /**
   * @brief Indexing operator
   *
   * @param y Index along the y-axis.
   * @param x Index along the x-axis (fastest changing index).
   * @return A const reference to the element at y,x.
   */
  const T& operator()(size_t y, size_t x) const {
    return this->m_buffer[index(y, x)];
  }

  /**
   * @brief Indexing operator
   *
   * @param y Index along the y-axis.
   * @param x Index along the x-axis (fastest changing index).
   * @return A reference to the element at y,x.
   */
  T& operator()(size_t y, size_t x) { return this->m_buffer[index(y, x)]; }

 protected:
  size_t m_y_dim;
  size_t m_x_dim;
};

template <class T>
class Array3D : public Array1D<T> {
 public:
  /**
   * @brief Construct a new Array3D object
   *
   * The memory is allocated through an auxiliary::AlignedMemory object
   *
   * @param z_dim Size of the z dimension
   * @param y_dim Size of the y dimension
   * @param x_dim Size of the x dimension (fastest changing index)
   */
  Array3D(size_t z_dim = 0, size_t y_dim = 0, size_t x_dim = 0)
      : Array1D<T>(z_dim * y_dim * x_dim),
        m_z_dim(z_dim),
        m_y_dim(y_dim),
        m_x_dim(x_dim) {}

  /**
   * @brief Construct a new Array3D object from a raw pointer
   *
   * @param data Pointer to the memory that the Array will use.
   *             Must be avaible during the life time of the Array
   *             The user is responsible for allocation and deallocation.
   * @param z_dim Size of the z dimension
   * @param y_dim Size of the y dimension
   * @param x_dim Size of the x dimension (fastest changing index)
   */
  Array3D(T* data, size_t z_dim, size_t y_dim, size_t x_dim)
      : Array1D<T>(data, z_dim * y_dim * x_dim),
        m_z_dim(z_dim),
        m_y_dim(y_dim),
        m_x_dim(x_dim) {}

  /**
   * @brief Construct a new Array3D object from a Memory object
   *
   * @param memory Unique pointer to the Memory object that will hold the data.
   *               Must be large enough to hold all elements.
   *               The Array class will take ownership of the unique_ptr by
   * moving it.
   * @param z_dim Size of the z dimension
   * @param y_dim Size of the y dimension
   * @param x_dim Size of the x dimension (fastest changing index)
   */
  Array3D(std::unique_ptr<auxiliary::Memory> memory, size_t z_dim, size_t y_dim,
          size_t x_dim)
      : Array1D<T>(std::move(memory), z_dim * y_dim * x_dim),
        m_z_dim(z_dim),
        m_y_dim(y_dim),
        m_x_dim(x_dim) {}

  /**
   * @brief Move constructor
   *
   * If the memory was managed by the other Array, the new
   * Array will take over ownership.
   *
   * @param other Pointer to the other Array
   */
  Array3D(Array3D&& other) {
    this->m_memory = std::move(other.m_memory);
    this->m_buffer = other.m_buffer;
    other.m_buffer = nullptr;
    m_z_dim = other.m_z_dim;
    m_y_dim = other.m_y_dim;
    m_x_dim = other.m_x_dim;
  }

  /**
   * @brief Move assignment operator
   *
   * If the memory was managed by the other Array, the new
   * Array will take over ownership.
   *
   * @param other Pointer to the other Array
   * @return Reference to the newly constructed Array
   */
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

  /**
   * @brief Get the length of the x dimension
   *
   * @return Length of the x dimension
   */
  size_t get_x_dim() const { return m_x_dim; }

  /**
   * @brief Get the length of the y dimension
   *
   * @return Length of the y dimension
   */
  size_t get_y_dim() const { return m_y_dim; }

  /**
   * @brief Get the length of the z dimension
   *
   * @return Length of the z dimension
   */
  size_t get_z_dim() const { return m_z_dim; }

  size_t size() const override { return m_z_dim * m_y_dim * m_x_dim; }

  /**
   * @brief Compute the linear index from multi dimensional indices
   *
   * @param z Index along the z-axis
   * @param y Index along the y-axis
   * @param x Index along the x-axis (fastest changing index)
   * @return linear index of element at z,y,x
   */
  inline size_t index(size_t z, size_t y, size_t x) const {
    return z * m_y_dim * m_x_dim + y * m_x_dim + x;
  }

  /**
   * @brief Get a pointer to the data at position z, y, x
   *
   * @param z Index along the z-axis
   * @param y Index along the y-axis
   * @param x Index along the x-axis (fastest changing index)
   * @return T*
   */
  T* data(size_t z = 0, size_t y = 0, size_t x = 0) const {
    return &this->m_buffer[index(z, y, x)];
  }

  /**
   * @brief Indexing operator
   *
   * @param z Index along the z-axis.
   * @param y Index along the y-axis.
   * @param x Index along the x-axis (fastest changing index).
   * @return A const reference to the element at z,y,x.
   */
  const T& operator()(size_t z, size_t y, size_t x) const {
    return this->m_buffer[index(z, y, x)];
  }

  /**
   * @brief Indexing operator
   *
   * @param z Index along the z-axis.
   * @param y Index along the y-axis.
   * @param x Index along the x-axis (fastest changing index).
   * @return A reference to the element at z,y,x.
   */
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
  /**
   * @brief Construct a new Array4D object
   *
   * The memory is allocated through an auxiliary::AlignedMemory object
   *
   * @param w_dim Size of the w dimension
   * @param z_dim Size of the z dimension
   * @param y_dim Size of the y dimension
   * @param x_dim Size of the x dimension (fastest changing index)
   */
  Array4D(size_t w_dim = 0, size_t z_dim = 0, size_t y_dim = 0,
          size_t x_dim = 0)
      : Array1D<T>(w_dim * z_dim * y_dim * x_dim),
        m_w_dim(w_dim),
        m_z_dim(z_dim),
        m_y_dim(y_dim),
        m_x_dim(x_dim) {}

  /**
   * @brief Construct a new Array4D object from a raw pointer
   *
   * @param data Pointer to the memory that the Array will use.
   *             Must be avaible during the life time of the Array
   *             The user is responsible for allocation and deallocation.
   * @param w_dim Size of the w dimension
   * @param z_dim Size of the z dimension
   * @param y_dim Size of the y dimension
   * @param x_dim Size of the x dimension (fastest changing index)
   */
  Array4D(T* data, size_t w_dim, size_t z_dim, size_t y_dim, size_t x_dim)
      : Array1D<T>(data, w_dim * z_dim * y_dim * x_dim),
        m_w_dim(w_dim),
        m_z_dim(z_dim),
        m_y_dim(y_dim),
        m_x_dim(x_dim) {}

  /**
   * @brief Construct a new Array4D object from a Memory object
   *
   * @param memory Unique pointer to the Memory object that will hold the data.
   *               Must be large enough to hold all elements.
   *               The Array class will take ownership of the unique_ptr by
   * moving it.
   * @param w_dim Size of the w dimension
   * @param z_dim Size of the z dimension
   * @param y_dim Size of the y dimension
   * @param x_dim Size of the x dimension (fastest changing index)
   */
  Array4D(std::unique_ptr<auxiliary::Memory> memory, size_t w_dim, size_t z_dim,
          size_t y_dim, size_t x_dim)
      : Array1D<T>(std::move(memory), w_dim * z_dim * y_dim * x_dim),
        m_w_dim(w_dim),
        m_z_dim(z_dim),
        m_y_dim(y_dim),
        m_x_dim(x_dim) {}

  /**
   * @brief Move constructor
   *
   * If the memory was managed by the other Array, the new
   * Array will take over ownership.
   *
   * @param other Pointer to the other Array
   */
  Array4D(Array4D&& other) {
    this->m_memory = std::move(other.m_memory);
    this->m_buffer = other.m_buffer;
    other.m_buffer = nullptr;
    m_w_dim = other.m_w_dim;
    m_z_dim = other.m_z_dim;
    m_y_dim = other.m_y_dim;
    m_x_dim = other.m_x_dim;
  }

  /**
   * @brief Move assignment operator
   *
   * If the memory was managed by the other Array, the new
   * Array will take over ownership.
   *
   * @param other Pointer to the other Array
   * @return Reference to the newly constructed Array\\
   */
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

  /**
   * @brief Get the length of the x dimension
   *
   * @return Length of the x dimension
   */
  size_t get_x_dim() const { return m_x_dim; }

  /**
   * @brief Get the length of the y dimension
   *
   * @return Length of the y dimension
   */
  size_t get_y_dim() const { return m_y_dim; }

  /**
   * @brief Get the length of the z dimension
   *
   * @return Length of the z dimension
   */
  size_t get_z_dim() const { return m_z_dim; }

  /**
   * @brief Get the length of the w dimension
   *
   * @return Length of the w dimension
   */
  size_t get_w_dim() const { return m_w_dim; }

  size_t size() const override { return m_w_dim * m_z_dim * m_y_dim * m_x_dim; }

  /**
   * @brief Compute the linear index from multi dimensional indices
   *
   * @param w Index along the w-axis
   * @param z Index along the z-axis
   * @param y Index along the y-axis
   * @param x Index along the x-axis (fastest changing index)
   * @return linear index of element at w,z,y,x
   */
  inline size_t index(size_t w, size_t z, size_t y, size_t x) const {
    return w * m_z_dim * m_y_dim * m_x_dim + z * m_y_dim * m_x_dim +
           y * m_x_dim + x;
  }

  /**
   * @brief Get a pointer to the data at position w, z, y, x
   *
   * @param w Index along the z-axis
   * @param z Index along the z-axis
   * @param y Index along the y-axis
   * @param x Index along the x-axis (fastest changing index)
   * @return T*
   */
  T* data(size_t w = 0, size_t z = 0, size_t y = 0, size_t x = 0) const {
    return &this->m_buffer[index(w, z, y, x)];
  }

  /**
   * @brief Indexing operator
   *
   * @param w Index along the w-axis.
   * @param z Index along the z-axis.
   * @param y Index along the y-axis.
   * @param x Index along the x-axis (fastest changing index).
   * @return A const reference to the element at w,z,y,x.
   */
  const T& operator()(size_t w, size_t z, size_t y, size_t x) const {
    return this->m_buffer[index(w, z, y, x)];
  }

  /**
   * @brief Indexing operator
   *
   * @param w Index along the w-axis.
   * @param z Index along the z-axis.
   * @param y Index along the y-axis.
   * @param x Index along the x-axis (fastest changing index).
   * @return A reference to the element at w,z,y,x.
   */
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
  /**
   * @brief Construct a new Array5D object
   *
   * The memory is allocated through an auxiliary::AlignedMemory object
   *
   * @param e_dim Size of the e dimension
   * @param d_dim Size of the d dimension
   * @param c_dim Size of the c dimension
   * @param b_dim Size of the b dimension
   * @param a_dim Size of the a dimension (fastest changing index)
   */
  Array5D(size_t e_dim = 0, size_t d_dim = 0, size_t c_dim = 0,
          size_t b_dim = 0, size_t a_dim = 0)
      : Array1D<T>(e_dim * d_dim * c_dim * b_dim * a_dim),
        m_e_dim(e_dim),
        m_d_dim(d_dim),
        m_c_dim(c_dim),
        m_b_dim(b_dim),
        m_a_dim(a_dim) {}

  /**
   * @brief Construct a new Array5D object from a raw pointer
   *
   * @param data Pointer to the memory that the Array will use.
   *             Must be avaible during the life time of the Array
   *             The user is responsible for allocation and deallocation.
   * @param e_dim Size of the e dimension
   * @param d_dim Size of the d dimension
   * @param c_dim Size of the c dimension
   * @param b_dim Size of the b dimension
   * @param a_dim Size of the a dimension (fastest changing index)
   */
  Array5D(T* data, size_t e_dim, size_t d_dim, size_t c_dim, size_t b_dim,
          size_t a_dim)
      : Array1D<T>(data, e_dim * d_dim * c_dim * b_dim * a_dim),
        m_e_dim(e_dim),
        m_d_dim(d_dim),
        m_c_dim(c_dim),
        m_b_dim(b_dim),
        m_a_dim(a_dim) {}

  /**
   * @brief Construct a new Array5D object from a Memory object
   *
   * @param memory Unique pointer to the Memory object that will hold the data.
   *               Must be large enough to hold all elements.
   *               The Array class will take ownership of the unique_ptr by
   * moving it.
   * @param e_dim Size of the e dimension
   * @param d_dim Size of the d dimension
   * @param c_dim Size of the c dimension
   * @param b_dim Size of the b dimension
   * @param a_dim Size of the a dimension (fastest changing index)
   */
  Array5D(std::unique_ptr<auxiliary::Memory> memory, size_t e_dim, size_t d_dim,
          size_t z_dim, size_t c_dim, size_t b_dim, size_t a_dim)
      : Array1D<T>(std::move(memory), e_dim * d_dim * c_dim * b_dim * a_dim),
        m_e_dim(e_dim),
        m_d_dim(d_dim),
        m_c_dim(c_dim),
        m_b_dim(b_dim),
        m_a_dim(a_dim) {}

  /**
   * @brief Move constructor
   *
   * If the memory was managed by the other Array, the new
   * Array will take over ownership.
   *
   * @param other Pointer to the other Array
   */
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

  /**
   * @brief Move assignment operator
   *
   * If the memory was managed by the other Array, the new
   * Array will take over ownership.
   *
   * @param other Pointer to the other Array
   * @return Reference to the newly constructed Array\\
   */
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

  /**
   * @brief Get the length of the a dimension
   *
   * @return Length of the a dimension
   */
  size_t get_a_dim() const { return m_a_dim; }

  /**
   * @brief Get the length of the b dimension
   *
   * @return Length of the b dimension
   */
  size_t get_b_dim() const { return m_b_dim; }

  /**
   * @brief Get the c dim object
   *
   * @return size_t
   */
  size_t get_c_dim() const { return m_c_dim; }

  /**
   * @brief Get the d dim object
   *
   * @return size_t
   */
  size_t get_d_dim() const { return m_d_dim; }

  /**
   * @brief Get the e dim object
   *
   * @return size_t
   */
  size_t get_e_dim() const { return m_e_dim; }

  size_t size() const override {
    return m_e_dim * m_d_dim * m_c_dim * m_b_dim * m_a_dim;
  }

  /**
   * @brief Compute the linear index from multi dimensional indices
   *
   * @param e Index along the e-axis
   * @param d Index along the d-axis
   * @param c Index along the c-axis
   * @param b Index along the b-axis
   * @param a Index along the a-axis (fastest changing index)
   * @return linear index of element at e,d,c,b,a
   */
  inline size_t index(size_t e, size_t d, size_t c, size_t b, size_t a) const {
    return e * m_d_dim * m_c_dim * m_b_dim * m_a_dim +
           d * m_c_dim * m_b_dim * m_a_dim + c * m_b_dim * m_a_dim +
           b * m_a_dim + a;
  }

  /**
   * @brief Get a pointer to the data at position e, d, c, b, a
   *
   * @param e Index along the e-axis
   * @param d Index along the d-axis
   * @param c Index along the c-axis
   * @param b Index along the b-axis
   * @param a Index along the a-axis (fastest changing index)
   * @return T*
   */
  T* data(size_t e = 0, size_t d = 0, size_t c = 0, size_t b = 0,
          size_t a = 0) const {
    return &this->m_buffer[index(e, d, c, b, a)];
  }

  /**
   * @brief Indexing operator
   *
   * @param e Index along the e-axis.
   * @param d Index along the d-axis.
   * @param c Index along the c-axis.
   * @param b Index along the b-axis.
   * @param a Index along the a-axis (fastest changing index).
   * @return A const reference to the element at e,d,c,b,a.
   */
  const T& operator()(size_t e = 0, size_t d = 0, size_t c = 0, size_t b = 0,
                      size_t a = 0) const {
    return this->m_buffer[index(e, d, c, b, a)];
  }

  /**
   * @brief Indexing operator
   *
   * @param e Index along the e-axis.
   * @param d Index along the d-axis.
   * @param c Index along the c-axis.
   * @param b Index along the b-axis.
   * @param a Index along the a-axis (fastest changing index).
   * @return A reference to the element at e,d,c,b,a.
   */
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

template <class T>
class Array6D : public Array1D<T> {
 public:
  /**
   * @brief Construct a new Array6D object
   *
   * The memory is allocated through an auxiliary::AlignedMemory object
   *
   * @param f_dim Size of the f dimension
   * @param e_dim Size of the e dimension
   * @param d_dim Size of the d dimension
   * @param c_dim Size of the c dimension
   * @param b_dim Size of the b dimension
   * @param a_dim Size of the a dimension (fastest changing index)
   */
  Array6D(size_t f_dim = 0, size_t e_dim = 0, size_t d_dim = 0,
          size_t c_dim = 0, size_t b_dim = 0, size_t a_dim = 0)
      : Array1D<T>(f_dim * e_dim * d_dim * c_dim * b_dim * a_dim),
        m_f_dim(e_dim),
        m_e_dim(e_dim),
        m_d_dim(d_dim),
        m_c_dim(c_dim),
        m_b_dim(b_dim),
        m_a_dim(a_dim) {}

  /**
   * @brief Construct a new Array6D object from a raw pointer
   *
   * @param data Pointer to the memory that the Array will use.
   *             Must be avaible during the life time of the Array
   *             The user is responsible for allocation and deallocation.
   * @param f_dim Size of the f dimension
   * @param e_dim Size of the e dimension
   * @param d_dim Size of the d dimension
   * @param c_dim Size of the c dimension
   * @param b_dim Size of the b dimension
   * @param a_dim Size of the a dimension (fastest changing index)
   */
  Array6D(T* data, size_t f_dim, size_t e_dim, size_t d_dim, size_t c_dim,
          size_t b_dim, size_t a_dim)
      : Array1D<T>(data, f_dim * e_dim * d_dim * c_dim * b_dim * a_dim),
        m_f_dim(f_dim),
        m_e_dim(e_dim),
        m_d_dim(d_dim),
        m_c_dim(c_dim),
        m_b_dim(b_dim),
        m_a_dim(a_dim) {}

  /**
   * @brief Construct a new Array6D object from a Memory object
   *
   * @param memory Unique pointer to the Memory object that will hold the data.
   *               Must be large enough to hold all elements.
   *               The Array class will take ownership of the unique_ptr by
   * moving it.
   * @param f_dim Size of the f dimension
   * @param e_dim Size of the e dimension
   * @param d_dim Size of the d dimension
   * @param c_dim Size of the c dimension
   * @param b_dim Size of the b dimension
   * @param a_dim Size of the a dimension (fastest changing index)
   */
  Array6D(std::unique_ptr<auxiliary::Memory> memory, size_t f_dim, size_t e_dim,
          size_t d_dim, size_t z_dim, size_t c_dim, size_t b_dim, size_t a_dim)
      : Array1D<T>(std::move(memory),
                   f_dim * e_dim * d_dim * c_dim * b_dim * a_dim),
        m_f_dim(f_dim),
        m_e_dim(e_dim),
        m_d_dim(d_dim),
        m_c_dim(c_dim),
        m_b_dim(b_dim),
        m_a_dim(a_dim) {}

  /**
   * @brief Move constructor
   *
   * If the memory was managed by the other Array, the new
   * Array will take over ownership.
   *
   * @param other Pointer to the other Array
   */
  Array6D(Array6D&& other) {
    this->m_memory = std::move(other.m_memory);
    this->m_buffer = other.m_buffer;
    other.m_buffer = nullptr;
    m_f_dim = other.m_f_dim;
    m_e_dim = other.m_e_dim;
    m_d_dim = other.m_d_dim;
    m_c_dim = other.m_c_dim;
    m_b_dim = other.m_b_dim;
    m_a_dim = other.m_a_dim;
  }

  /**
   * @brief Move assignment operator
   *
   * If the memory was managed by the other Array, the new
   * Array will take over ownership.
   *
   * @param other Pointer to the other Array
   * @return Reference to the newly constructed Array\\
   */
  Array6D& operator=(Array6D&& other) {
    this->m_memory = std::move(other.m_memory);
    this->m_buffer = other.m_buffer;
    other.m_buffer = nullptr;
    m_f_dim = other.m_f_dim;
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
    m_f_dim = 0;
  }

  /**
   * @brief Get the length of the a dimension
   *
   * @return Length of the a dimension (fastest chaning index)
   */
  size_t get_a_dim() const { return m_a_dim; }

  /**
   * @brief Get the length of the b dimension
   *
   * @return Length of the b dimension
   */
  size_t get_b_dim() const { return m_b_dim; }

  /**
   * @brief Get the c dim object
   *
   * @return size_t
   */
  size_t get_c_dim() const { return m_c_dim; }

  /**
   * @brief Get the d dim object
   *
   * @return size_t
   */
  size_t get_d_dim() const { return m_d_dim; }

  /**
   * @brief Get the e dim object
   *
   * @return size_t
   */
  size_t get_e_dim() const { return m_e_dim; }

  /**
   * @brief Get the f dim object
   *
   * @return size_t
   */
  size_t get_f_dim() const { return m_f_dim; }

  size_t size() const override {
    return m_f_dim * m_e_dim * m_d_dim * m_c_dim * m_b_dim * m_a_dim;
  }

  /**
   * @brief Compute the linear index from multi dimensional indices
   *
   * @param f Index along the f-axis
   * @param e Index along the e-axis
   * @param d Index along the d-axis
   * @param c Index along the c-axis
   * @param b Index along the b-axis
   * @param a Index along the a-axis (fastest changing index)
   * @return linear index of element at position (f,e,d,c,b,a)
   */
  inline size_t index(size_t f, size_t e, size_t d, size_t c, size_t b,
                      size_t a) const {
    return f * m_e_dim * m_d_dim * m_c_dim * m_b_dim * m_a_dim +
           e * m_d_dim * m_c_dim * m_b_dim * m_a_dim +
           d * m_c_dim * m_b_dim * m_a_dim + c * m_b_dim * m_a_dim +
           b * m_a_dim + a;
  }

  /**
   * @brief Get a pointer to the data at position f, e, d, c, b, a
   *
   * @param f Index along the e-axis
   * @param e Index along the e-axis
   * @param d Index along the d-axis
   * @param c Index along the c-axis
   * @param b Index along the b-axis
   * @param a Index along the a-axis (fastest changing index)
   * @return T*
   */
  T* data(size_t f = 0, size_t e = 0, size_t d = 0, size_t c = 0, size_t b = 0,
          size_t a = 0) const {
    return &this->m_buffer[index(f, e, d, c, b, a)];
  }

  /**
   * @brief Indexing operator
   *
   * @param f Index along the f-axis.
   * @param e Index along the e-axis.
   * @param d Index along the d-axis.
   * @param c Index along the c-axis.
   * @param b Index along the b-axis.
   * @param a Index along the a-axis (fastest changing index).
   * @return A const reference to the element at position (f,e,d,c,b,a).
   */
  const T& operator()(size_t f = 0, size_t e = 0, size_t d = 0, size_t c = 0,
                      size_t b = 0, size_t a = 0) const {
    return this->m_buffer[index(f, e, d, c, b, a)];
  }

  /**
   * @brief Indexing operator
   *
   * @param f Index along the f-axis.
   * @param e Index along the e-axis.
   * @param d Index along the d-axis.
   * @param c Index along the c-axis.
   * @param b Index along the b-axis.
   * @param a Index along the a-axis (fastest changing index).
   * @return A reference to the element at e,d,c,b,a.
   */
  T& operator()(size_t f = 0, size_t e = 0, size_t d = 0, size_t c = 0,
                size_t b = 0, size_t a = 0) {
    return this->m_buffer[index(f, e, d, c, b, a)];
  }

 protected:
  size_t m_f_dim;
  size_t m_e_dim;
  size_t m_d_dim;
  size_t m_c_dim;
  size_t m_b_dim;
  size_t m_a_dim;
};

using Grid = Array4D<std::complex<float>>;

template <class T>
std::ostream& operator<<(std::ostream& os, const Array1D<T>& a) {
  for (size_t x = 0; x < a.get_x_dim(); ++x) {
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
  for (size_t y = 0; y < a.get_y_dim(); ++y) {
    for (size_t x = 0; x < a.get_x_dim(); ++x) {
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
  for (size_t z = 0; z < a.get_z_dim(); ++z) {
    os << std::endl;
    for (size_t y = 0; y < a.get_y_dim(); ++y) {
      for (size_t x = 0; x < a.get_x_dim(); ++x) {
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
  for (size_t w = 0; w < a.get_w_dim(); ++w) {
    os << std::endl;
    for (size_t z = 0; z < a.get_z_dim(); ++z) {
      os << std::endl;
      for (size_t y = 0; y < a.get_y_dim(); ++y) {
        for (size_t x = 0; x < a.get_x_dim(); ++x) {
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
