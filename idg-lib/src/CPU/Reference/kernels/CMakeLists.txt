project(cpu-reference-kernel)

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
endif()

# Compiler options
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    add_compile_options("-qopenmp")
    add_compile_options("-mkl")
else()
    add_compile_options("-fopenmp")
    if (BUILD_WITH_MKL)
        link_libraries(${MKL_LIBRARIES})
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

SET_TARGET_PROPERTIES(${PROJECT_NAME}-gridder PROPERTIES SOVERSION 0)
SET_TARGET_PROPERTIES(${PROJECT_NAME}-degridder PROPERTIES SOVERSION 0)
SET_TARGET_PROPERTIES(${PROJECT_NAME}-adder PROPERTIES SOVERSION 0)
SET_TARGET_PROPERTIES(${PROJECT_NAME}-splitter PROPERTIES SOVERSION 0)
SET_TARGET_PROPERTIES(${PROJECT_NAME}-fft PROPERTIES SOVERSION 0)

# Copy kernel libraries
install(
    TARGETS
    ${PROJECT_NAME}-gridder
    ${PROJECT_NAME}-degridder
    ${PROJECT_NAME}-adder
    ${PROJECT_NAME}-splitter
    ${PROJECT_NAME}-fft
    DESTINATION
    lib/idg-cpu/Reference
)
