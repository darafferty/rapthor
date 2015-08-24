cmake_minimum_required(VERSION 2.8.11)

# the project
project(ImageDomainGridder)
set (VERSION_MAJOR 0)
set (VERSION_MINOR 1)
set (VERSION_PATCH 0)
set (VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")

# Set compile flags 
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -fopenmp -g")

# add subfolders
add_subdirectory(src)

