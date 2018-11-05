# - Try to find clFFT
# This module tries to find the clFFT library on your system
#
# Once done this will define
#  CLFFT_FOUND        - system has clFFT
#  CLFFT_INCLUDE_DIRS - the clFFT include directory
#  CLFFT_LIBRARIES    - link these to use clFFT

find_package(PackageHandleStandardArgs)

# Find libclFFT.so
find_library(CLFFT_LIBRARIES clFFT
    ENV CLFFT_LIB
    ENV LD_LIBRARY_PATH
)
get_filename_component(CLFFT_LIB_DIR ${CLFFT_LIBRARIES} PATH)

# Look for clFFT header files in default location (relative to library)
get_filename_component(_CLFFT_INCLUDE_DIR ${CLFFT_LIB_DIR}/../include ABSOLUTE)
find_path(CLFFT_INCLUDE_DIRS clFFT.h PATHS ${_CLFFT_INCLUDE_DIR})

find_package_handle_standard_args(clFFT DEFAULT_MSG CLFFT_LIBRARIES CLFFT_INCLUDE_DIRS)
