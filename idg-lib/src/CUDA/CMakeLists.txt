project(cuda)

# resolve dependencies
find_package(CUDA)
find_package(CUDAdriver)
if (${CUDA_FOUND})
mark_as_advanced (CUDA_TOOLKIT_ROOT_DIR)
mark_as_advanced (CUDA_BUILD_CUBIN)
mark_as_advanced (CUDA_BUILD_EMULATION)
mark_as_advanced (CUDA_HOST_COMPILER)
mark_as_advanced (CUDA_SDK_ROOT_DIR)
mark_as_advanced (CUDA_SEPARABLE_COMPILATION)
mark_as_advanced (CUDA_VERBOSE_BUILD)
set (CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_INCLUDE}" CACHE PATH "CUDA include directories")
set (CUDA_FFT_LIBRARY "${CUDA_cufft_LIBRARY}" CACHE PATH "CUDA fft library")
endif ()

# add subfolders
add_subdirectory(common)
add_subdirectory(Generic)
