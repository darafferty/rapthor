# adjust the default behaviour of the FIND_XXX() commands:
# search headers and libraries in the target environment, search
# programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# The serial compilers
set(CMAKE_C_COMPILER       "/cm/shared/package/gcc/6.1.0/bin/gcc")
set(CMAKE_CXX_COMPILER     "/cm/shared/package/gcc/6.1.0/bin/g++")

# Set CUDA environment
set(CUDA_TOOLKIT_ROOT_DIR "/cm/shared/apps/cuda75/toolkit/7.5.18")

# Set MKL environment (always used, independent of BUILD_WITH_MKL)
# set(MKL_INCLUDE_DIRS "/cm/shared/package/2016.3/mkl/include")
# set(MKL_LIBRARIES "/cm/shared/package/2016.3/mkl/lib/intel64/libmkl_rt.so")
# set(FFTW3_INCLUDE_DIR "/cm/shared/package/intel/2016.3/mkl/include/fftw")
# set(FFTW3F_LIBRARY ${MKL_LIBRARIES})
# set(FFTW3_LIBRARY ${MKL_LIBRARIES})
