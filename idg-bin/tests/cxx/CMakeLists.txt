project(cxx)

# set utility include directory
include_directories(${IDG_BIN_UTILITY_INCLUDE_DIR})

# set common include directory
include_directories(common)

# set common link libraries
set(LINK_LIBRARIES idg-util idg-cpu)

# add subfolders
if (BUILD_LIB_CPU)
    add_subdirectory(CPU)
endif()
add_subdirectory(Hybrid)
if (BUILD_LIB_CUDA)
    add_subdirectory(CUDA)
endif()
if (BUILD_LIB_OPENCL)
    add_subdirectory(OpenCL)
endif()
