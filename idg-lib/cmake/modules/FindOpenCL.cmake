# - Try to find OpenCL
# This module tries to find an OpenCL implementation on your system.
# It supports AMD and NVIDIA implementations.
#
# It looks for OpenCL libraries in your LD_LIBRARY_PATH and tries
# to resolve the include directory relative to the libraries.
# As an alternative, specify OPENCL_LIB and OPENCL_INCLUDE manually
#
# Once done this will define
#  OPENCL_FOUND        - system has OpenCL
#  OPENCL_INCLUDE_DIRS - the OpenCL include directory
#  OPENCL_LIBRARIES    - link these to use OpenCL

find_package(PackageHandleStandardArgs)

# Find libOpenCL.so
find_library(OPENCL_LIBRARIES
    NAMES OpenCL
    HINTS ENV OPENCL_LIB
    HINTS ENV LD_LIBRARY_PATH
)
get_filename_component(OPENCL_LIB_DIR ${OPENCL_LIBRARIES} PATH)

# Look for OpenCL header files in default location (relative to library)
get_filename_component(_OPENCL_INCLUDE_DIR ${OPENCL_LIB_DIR}/../../include ABSOLUTE)
find_path(OPENCL_INCLUDE_DIRS CL/cl.h PATHS ${_OPENCL_INCLUDE_DIR})
set (_OPENCL_CPP_INCLUDE_DIRS INTERNAL)
find_path(_OPENCL_CPP_INCLUDE_DIRS CL/cl.hpp PATHS ${_OPENCL_INCLUDE_DIR})

# Look for OpenCL header files in $OPENCL_INCLUDE directory
set(_OPENCL_INCLUDE_DIR $ENV{OPENCL_INCLUDE})
find_path(OPENCL_INCLUDE_DIRS CL/cl.h PATHS ${_OPENCL_INCLUDE_DIR})
find_path(_OPENCL_CPP_INCLUDE_DIRS CL/cl.hpp PATHS ${_OPENCL_INCLUDE_DIR})

if(_OPENCL_CPP_INCLUDE_DIRS)
    # Add include directory for cpp bindings
	list(APPEND OPENCL_INCLUDE_DIRS ${_OPENCL_CPP_INCLUDE_DIRS})

	# This is often the same, so clean up
	list( REMOVE_DUPLICATES OPENCL_INCLUDE_DIRS )
endif(_OPENCL_CPP_INCLUDE_DIRS)

find_package_handle_standard_args(OpenCL DEFAULT_MSG OPENCL_LIBRARIES OPENCL_INCLUDE_DIRS)
