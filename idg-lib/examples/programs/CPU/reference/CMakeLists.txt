cmake_minimum_required(VERSION 2.8.11)

project(ImageDomainGridder)

# Set compile flags
set(CMAKE_CXX_FLAGS "-std=c++0x -Wall -O2 -fopenmp -DREPORT_VERBOSE -g")

set(UTILITY_DIR ../../../utility)
set(EXTERNAL_DIR ../../../external)

# Set header files
include_directories(
    ${IDG_ROOT_DIR}/include
    ${UTILITY_DIR}
    ${EXTERNAL_DIR}
)

set(
    HEADERS
    ${UTILITY_DIR}/Init.h
    ${EXTERNAL_DIR}/uvwsim.h
)

set(
    SOURCES
    main.cpp
    ${UTILITY_DIR}/Init.cpp
    ${EXTERNAL_DIR}/uvwsim.cpp
)

# Set build target
add_executable(main ${SOURCES} ${HEADERS})

# Set library dependencies
include_directories(${IDG_ROOT_DIR}/lib)
target_link_libraries(main idg-cpu-reference)

find_package(LIBDL)
target_link_libraries(main ${DL_LIBRARIES})
