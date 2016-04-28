# - Try to find likwid
# This module tries to find the likwid library on your system
#
# Once done this will define
#  LIKWID_FOUND        - system has likwid
#  LIKWID_INCLUDE_DIRS - the likwid include directory
#  LIKWID_LIBRARIES    - link these to use likwid

FIND_PACKAGE( PackageHandleStandardArgs )

# Unix style platforms
FIND_LIBRARY(LIKWID_LIBRARIES likwid
    ENV LD_LIBRARY_PATH
)

GET_FILENAME_COMPONENT(LIKWID_LIB_DIR ${LIKWID_LIBRARIES} PATH)
GET_FILENAME_COMPONENT(_LIKWID_INC_DIR ${LIKWID_LIB_DIR}/../include ABSOLUTE)

# search for headers relative to the library
FIND_PATH(LIKWID_INCLUDE_DIRS likwid.h PATHS ${_LIKWID_INC_DIR})

FIND_PACKAGE_HANDLE_STANDARD_ARGS( likwid DEFAULT_MSG LIKWID_LIBRARIES LIKWID_INCLUDE_DIRS )
