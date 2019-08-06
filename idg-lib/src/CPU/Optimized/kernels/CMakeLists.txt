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
    link_libraries(${MKL_LIBRARIES})
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
        if (${USE_LOOKUP_TABLE})
            add_compile_options("-DUSE_LOOKUP")
        endif()
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
add_library(${PROJECT_NAME}-calibrate KernelCalibrate.cpp)

# Add libraries for w-stacking
add_library(${PROJECT_NAME}-adder-wstack     KernelAdderWStack.cpp)
add_library(${PROJECT_NAME}-splitter-wstack  KernelSplitterWStack.cpp)

# Add libraries for w-tiles
add_library(${PROJECT_NAME}-adder-wtiles     KernelAdderWTiles.cpp)
add_library(${PROJECT_NAME}-splitter-wtiles  KernelSplitterWTiles.cpp)

target_link_libraries(${PROJECT_NAME}-adder-wtiles idg-fft)
target_link_libraries(${PROJECT_NAME}-splitter-wtiles idg-fft)

SET_TARGET_PROPERTIES(${PROJECT_NAME}-gridder PROPERTIES SOVERSION 0)
SET_TARGET_PROPERTIES(${PROJECT_NAME}-degridder PROPERTIES SOVERSION 0)
SET_TARGET_PROPERTIES(${PROJECT_NAME}-adder PROPERTIES SOVERSION 0)
SET_TARGET_PROPERTIES(${PROJECT_NAME}-splitter PROPERTIES SOVERSION 0)
SET_TARGET_PROPERTIES(${PROJECT_NAME}-fft PROPERTIES SOVERSION 0)
SET_TARGET_PROPERTIES(${PROJECT_NAME}-calibrate PROPERTIES SOVERSION 0)
SET_TARGET_PROPERTIES(${PROJECT_NAME}-adder-wstack PROPERTIES SOVERSION 0)
SET_TARGET_PROPERTIES(${PROJECT_NAME}-splitter-wstack PROPERTIES SOVERSION 0)
SET_TARGET_PROPERTIES(${PROJECT_NAME}-adder-wtiles PROPERTIES SOVERSION 0)
SET_TARGET_PROPERTIES(${PROJECT_NAME}-splitter-wtiles PROPERTIES SOVERSION 0)

# Copy kernel libraries
install(
    TARGETS
    ${PROJECT_NAME}-gridder
    ${PROJECT_NAME}-degridder
    ${PROJECT_NAME}-adder
    ${PROJECT_NAME}-splitter
    ${PROJECT_NAME}-fft
    ${PROJECT_NAME}-calibrate
    ${PROJECT_NAME}-adder-wstack
    ${PROJECT_NAME}-splitter-wstack
    ${PROJECT_NAME}-adder-wtiles
    ${PROJECT_NAME}-splitter-wtiles
    DESTINATION
    lib/idg-cpu/Optimized
)
