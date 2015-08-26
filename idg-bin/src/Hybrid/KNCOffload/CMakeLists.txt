cmake_minimum_required(VERSION 2.8.11)

# set cmake policy
cmake_policy(SET CMP0022 OLD)

project(idg-hybrid-knc)

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

# external header files needed 
include_directories (
  ${CMAKE_SOURCE_DIR}/src/common
  ${CMAKE_SOURCE_DIR}/src/CPU/reference
)

# sources and header files
set (${PROJECT_NAME}_headers 
  KNCOffload.h
)

set (${PROJECT_NAME}_sources 
  KNCOffload.cpp
)

# create library
# add_library(
#  ${PROJECT_NAME} SHARED ${${PROJECT_NAME}_headers} 
#  ${${PROJECT_NAME}_sources}
#  )
add_library(
  ${PROJECT_NAME} ${${PROJECT_NAME}_headers} 
                  ${${PROJECT_NAME}_sources}
  )

# linker
target_link_libraries(${PROJECT_NAME} idg-hybrid-knc)

# Set installation directory
install(TARGETS ${PROJECT_NAME} 
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
  )

# Copy kernel source files
install(
    FILES
    kernels/KernelAdder.cpp
    kernels/KernelDegridder.cpp
    kernels/KernelFFT.cpp
    kernels/KernelGridder.cpp
    kernels/KernelSplitter.cpp
    DESTINATION
    lib/kernels/Hybrid/KNCOffload
)

# Copy header files
install(
    FILES
    idg.h
    KNCOffload.h
    DESTINATION
    include/Hybrid/KNCOffload
)
