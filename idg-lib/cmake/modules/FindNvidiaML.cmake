# Module for locating libnvidia-ml
#
# Read-only variables:
#   NVML_FOUND
#     Indicates that the library has been found.
#
#   NVML_INCLUDE_DIR
#     Points to the libnvidia-ml include directory.
#
#   NVML_LIBRARY_DIR
#     Points to the directory that contains the libraries.
#     The content of this variable can be passed to link_directories.
#
#   NVML_LIBRARY
#     Points to the libnvidia-ml that can be passed to target_link_libararies.

include(FindPackageHandleStandardArgs)

find_path(NVML_INCLUDE_DIR
  NAMES nvml.h
  HINTS ${NVML_ROOT_DIR} /usr/local/cuda
  ENV NVML_INCLUDE
  PATH_SUFFIX include
  DOC "NVML include directory")

find_library(NVML_LIBRARY
  NAMES nvidia-ml
  HINTS ${NVML_ROOT_DIR}
  ENV NVML_LIB
  ENV LD_LIBRARY_PATH
  DOC "NVML library")

if (NVML_LIBRARY)
    get_filename_component(NVML_LIBRARY_DIR ${NVML_LIBRARY} PATH)
endif()

mark_as_advanced(NVML_ROOT_DIR NVML_INCLUDE_DIR NVML_LIBRARY_DIR NVML_LIBRARY)

find_package_handle_standard_args(NVML REQUIRED_VARS NVML_INCLUDE_DIR NVML_LIBRARY)
