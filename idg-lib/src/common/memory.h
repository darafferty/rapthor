#include <stddef.h>

#include <iostream>

#define ALIGNMENT 64

template<class T>
T* allocate_memory(size_t n, unsigned int alignment = ALIGNMENT) {
    void *ptr = nullptr;
    if (n > 0) {
        size_t bytes = n * sizeof(T);
        bytes = (((bytes - 1) / alignment) * alignment) + alignment;
        if (posix_memalign(&ptr, alignment, bytes) != 0) {
            std::cerr << "Could not allocate " << bytes << " bytes" << std::endl;
            exit(EXIT_FAILURE);
        };
    }
    return (T *) ptr;
}