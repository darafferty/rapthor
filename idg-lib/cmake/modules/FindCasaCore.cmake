# Find casacore
# see: https://github.com/ska-sa/meqtrees-timba/blob/master/cmake/FindCasaCore.cmake
#
# Find the native CASACORE includes and library
#
#  CASACORE_INCLUDE_DIR  - where to find casacore.h, etc.
#  CASACORE_LIBRARIES    - List of libraries when using casacore.
#  CASACORE_FOUND        - True if casacore found.

IF (CASACORE_INCLUDE_DIR)
    # Already in cache, be silent
    SET(CASACORE_FIND_QUIETLY TRUE)
ENDIF (CASACORE_INCLUDE_DIR)

FIND_PATH(CASACORE_INCLUDE_DIR casacore HINT ENV CASACORE_INCLUDE)

SET(CASACORE_NAMES
    casa_images
    casa_mirlib
    casa_coordinates
    casa_lattices
    casa_msfits
    casa_ms
    casa_fits
    casa_measures
    casa_tables
    casa_scimath
    casa_scimath_f
    casa_casa
    casa_images
)
FOREACH( lib ${CASACORE_NAMES} )
    FIND_LIBRARY(CASACORE_LIBRARY_${lib} NAMES ${lib} PATHS ENV CASACORE_LIB )
    MARK_AS_ADVANCED(CASACORE_LIBRARY_${lib})
    LIST(APPEND CASACORE_LIBRARIES ${CASACORE_LIBRARY_${lib}})
ENDFOREACH(lib)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CASACORE DEFAULT_MSG CASACORE_LIBRARIES CASACORE_INCLUDE_DIR)

IF(CASACORE_FOUND)
ELSE(CASACORE_FOUND)
    SET( CASACORE_LIBRARIES )
ENDIF(CASACORE_FOUND)
