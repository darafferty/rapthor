cmake_minimum_required(VERSION 2.6)

project(ImageDomainGridder)

# Library will by default by installed in ../lib
set(CMAKE_INSTALL_PREFIX ..)

# Set compile flags
set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS} -Wall -O2 -fopenmp -DREPORT_VERBOSE")

# Set header files
include_directories(
    include
    ../../../common
    ../../../utility
)

# Library consists of interface to routines (Proxy.cpp) and interface to kernels (Kernels.cpp)
add_library(idg-shared SHARED Proxy.cpp Kernels.cpp)
add_library(idg-static STATIC Proxy.cpp Kernels.cpp)

# Name both files libidg.*
set_target_properties(idg-shared PROPERTIES OUTPUT_NAME idg-cpu-reference)
set_target_properties(idg-static PROPERTIES OUTPUT_NAME idg-cpu-reference)

# Set installation directory
install(TARGETS idg-shared DESTINATION lib)
install(TARGETS idg-static DESTINATION lib)

# Install as a library
install(
FILES
    Proxy.h Types.h
    Kernels.cpp Kernels.h
    KernelGridder.cpp
    KernelDegridder.cpp
    KernelFFT.cpp
    KernelAdder.cpp
    KernelSplitter.cpp
DESTINATION
    lib/idg/CPU/reference)
