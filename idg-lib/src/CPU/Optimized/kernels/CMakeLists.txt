project(cpu-optimized-kernel)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/
    ${FFTW3_INCLUDE_DIR}
)

# Build type options
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    add_compile_options("-Wall")
    add_compile_options("-DDEBUG")
else()
    add_compile_options("-O3")
    add_compile_options("-ffast-math")
endif()


# Compiler options
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    add_compile_options("-x${INSTRUCTION_SET}")
    add_compile_options("-DUSE_VML")
    add_compile_options("-mkl")
else()
    if (POWER_ARCHITECTURE)
        add_compile_options("-mcpu=native")
    else()
        add_compile_options("-march=native")
    endif()
    add_compile_options("-fopenmp")
    if (BUILD_WITH_MKL)
        add_compile_options("-DUSE_VML")
        add_compile_options("-lmkl")
        include_directories(${MKL_INCLUDE_DIRS})
    else()
        link_libraries(${FFTW3_LIBRARIES})
    endif()
endif()

# Miscellaneous options
add_compile_options("-DNR_POLARIZATIONS=4")

# Add libraries
add_library(${PROJECT_NAME}-gridder   KernelGridder.cpp)
add_library(${PROJECT_NAME}-degridder KernelDegridder.cpp)
add_library(${PROJECT_NAME}-adder     KernelAdder.cpp)
add_library(${PROJECT_NAME}-splitter  KernelSplitter.cpp)
add_library(${PROJECT_NAME}-fft       KernelFFT.cpp)

# Add libraries for w-stacking
add_library(${PROJECT_NAME}-adder-wstack     KernelAdderWStack.cpp)
add_library(${PROJECT_NAME}-splitter-wstack  KernelSplitterWStack.cpp)

# Copy kernel libraries
install(
    TARGETS
    ${PROJECT_NAME}-gridder
    ${PROJECT_NAME}-degridder
    ${PROJECT_NAME}-adder
    ${PROJECT_NAME}-splitter
    ${PROJECT_NAME}-fft
    ${PROJECT_NAME}-adder-wstack
    ${PROJECT_NAME}-splitter-wstack
    DESTINATION
    lib/idg-cpu/Optimized
)
