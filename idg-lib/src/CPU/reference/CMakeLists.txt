cmake_minimum_required(VERSION 2.8.11)

project(ImageDomainGridder)

# Set compile flags
set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS} -Wall -O2 -fopenmp -DREPORT_VERBOSE -g -fPIC")

# Set header files
set(COMMON_DIR ../../common)
#message("Common dir is=" ${COMMON_DIR})
include_directories(
    ${COMMON_DIR}
    ${COMMON_DIR}/utility
)

# Library consists of interface to routines (Proxy.cpp) and interface to kernels (Kernels.cpp)
set(
    HEADERS
    ${COMMON_DIR}/AbstractProxy.h  
    ${COMMON_DIR}/Parameters.h
    ${COMMON_DIR}/RuntimeWrapper.h
    ${COMMON_DIR}/CompilerEnvironment.h
    ${COMMON_DIR}/ProxyInfo.h
    ${COMMON_DIR}/utility/auxiliary.h
)

set(
    SOURCES
    CPU.cpp
    Kernels.cpp
    ${COMMON_DIR}/utility/auxiliary.cpp
)

add_library(idg-cpu-reference SHARED ${SOURCES} ${HEADERS})

# Include common directory as a dependency
add_subdirectory(${COMMON_DIR} ${COMMON_DIR})
target_link_libraries(idg-cpu-reference common)

# Set installation directory
install(TARGETS idg-cpu-reference DESTINATION lib)

# Install as a library
install(
    FILES
    kernels/KernelAdder.cpp
    kernels/KernelDegridder.cpp
    kernels/KernelFFT.cpp
    kernels/KernelGridder.cpp
    kernels/KernelSplitter.cpp
    DESTINATION
    kernels/CPU/reference
)

install(
    FILES
    ${HEADERS}
    CPU.h
    idg-cpu.h
    Types.h
    Kernels.h
    DESTINATION
    include/CPU/reference
)
