// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>

#define ALIGNMENT 64

template <class T>
T *allocate_memory(size_t n, unsigned int alignment = ALIGNMENT) {
  void *ptr = nullptr;
  if (n > 0) {
    size_t bytes = n * sizeof(T);
    bytes = (((bytes - 1) / alignment) * alignment) + alignment;

    // Try to allocate aligned memory
    auto status = posix_memalign(&ptr, alignment, bytes);

    if (status != 0) {
      std::stringstream message;
      message << "Could not posix_memalign " << bytes << " bytes";
      message << ", falling back to malloc";
      std::cerr << message.str() << std::endl;

      // Try again, using malloc
      ptr = malloc(bytes);

      if (!ptr) {
        std::stringstream message;
        message << "Could not malloc " << bytes << " bytes";
        throw std::runtime_error(message.str());
      }
    }
  }
  return (T *)ptr;
}