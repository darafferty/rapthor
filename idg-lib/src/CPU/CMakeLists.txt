project(cpu)

# resolve dependencies
if (BUILD_WITH_MKL AND NOT INTEL_CXX_COMPILER)
find_package(MKL QUIET)
endif()
if (${MKL_FOUND})
    get_filename_component(MKL_LIB_DIR ${MKL_LIBRARIES} PATH)
    set (HAVE_MKL TRUE)
    set(FFTW3F_LIBRARY ${MKL_LIBRARIES})
    set(FFTW3_LIBRARY ${MKL_LIBRARIES})
else()
    unset (MKL_INCLUDE_DIRS CACHE)
    unset (MKL_LIBRARIES CACHE)
    find_package(FFTW3 COMPONENTS single double)
endif()

# add subfolders
add_subdirectory(common)
add_subdirectory(Reference)
add_subdirectory(Optimized)
add_subdirectory(KNL)
