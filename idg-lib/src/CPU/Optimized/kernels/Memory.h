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