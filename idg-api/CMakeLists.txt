cmake_minimum_required(VERSION 2.8.0)

project(idg-wsclean)


option (BUILD_STATIC_LIBS "Build static libraries" OFF)

# build shared or static libraries (default: shared)
if (BUILD_STATIC_LIBS)
set (BUILD_SHARED_LIBS FALSE)
else()
set (BUILD_SHARED_LIBS TRUE)
endif()


# make sure C++11 is used (in newer CMake use CXX_STANDARD)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


# set cmake module path
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules")


find_package(IDG NO_MODULE)
include_directories (${IDG_INCLUDE_DIR})

find_package(LAPACKE)
include_directories (${LAPACK_INCLUDE_DIR})


# sources and header files
set (${PROJECT_NAME}_headers
    Buffer.h
    BufferSet.h
    GridderBuffer.h
    DegridderBuffer.h
    Datatypes.h
    taper.h
)

set (${PROJECT_NAME}_sources
    Buffer.cpp
    BufferSet.cpp
    GridderBuffer.cpp
    DegridderBuffer.cpp
    taper.cpp
)

set (${PROJECT_NAME}_link_libraries
    ${IDG_LIB}
    ${LAPACK_LIB}
)

add_library(
  ${PROJECT_NAME} ${${PROJECT_NAME}_headers}
                  ${${PROJECT_NAME}_sources}
)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-fPIC")
target_link_libraries (
    ${PROJECT_NAME} 
    ${${PROJECT_NAME}_link_libraries}
)



# Copy header files
install(
    FILES
    ${${PROJECT_NAME}_headers}
    DESTINATION
    include/idg-wsclean
)

install(
    FILES
    idg-wsclean.h
    DESTINATION
    include
)

install(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
  PUBLIC_HEADER DESTINATION include
)
