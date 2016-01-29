# - Try to find the Python Numpy header files and loadable modules.
# This find module tries to determine the directory path to the Numpy header
# files and the location of the loadable modules `multiarray` and `scalarmath`.
#
# This find module requires that Python is installed on the host system.
#
# The following variables will be set:
#  NUMPY_FOUND              - set to true if Numpy was found
#  NUMPY_PATH               - root directory of Numpy
#  NUMPY_INCLUDE_DIR        - directory with the Numpy header files (cached)
#  NUMPY_MULTIARRAY_LIBRARY - full path to the `multiarray` module (cached)
#  NUMPY_SCALARMATH_LIBRARY - full path to the `scalarmath` module (cached)
#  NUMPY_INCLUDE_DIRS       - list of include directories
#                             (identical to NUMPY_INCLUDE_DIR)
#  NUMPY_LIBRARIES          - list of Numpy loadable modules
#  F2PY_EXECUTABLE          - full path to the `f2py` program (cached)

if(NOT NUMPY_FOUND)

  if(NUMPY_FIND_REQUIRED)
    set(_required REQUIRED)
  endif(NUMPY_FIND_REQUIRED)
  if(NUMPY_FIND_QUIETLY)
    set(_quietly QUIETLY)
  endif(NUMPY_FIND_QUIETLY)
  if (NOT PYTHONINTERP_FOUND)
    find_package(PythonInterp ${_required} ${_quietly})
  endif (NOT PYTHONINTERP_FOUND)

  if(PYTHONINTERP_FOUND)
    # Temporarily set the library prefix to the empty string; python
    # module names do not have a "lib" prefix.
    set(_cmake_find_library_prefixes "${CMAKE_FIND_LIBRARY_PREFIXES}")
    set(CMAKE_FIND_LIBRARY_PREFIXES "")

    if(NOT NUMPY_PATH)
      set(_cmd "import numpy; print numpy.__path__[0]")
      execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" "-c" "${_cmd}"
        OUTPUT_VARIABLE _output
        RESULT_VARIABLE _result
        OUTPUT_STRIP_TRAILING_WHITESPACE)
      if(NOT ${_result} EQUAL 0)
        set(_output)
      endif(NOT ${_result} EQUAL 0)
      set(NUMPY_PATH "${_output}")
    endif(NOT NUMPY_PATH)

    if(NOT NUMPY_INCLUDE_DIR)
      set(_cmd "import numpy; print numpy.get_include()")
      execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" "-c" "${_cmd}"
        OUTPUT_VARIABLE _output
        RESULT_VARIABLE _result
        OUTPUT_STRIP_TRAILING_WHITESPACE)
      if(NOT ${_result} EQUAL 0)
        set(_numpy_include_dir)
      endif(NOT ${_result} EQUAL 0)
      set(NUMPY_INCLUDE_DIR ${_output} CACHE PATH "Numpy include directory")
    endif(NOT NUMPY_INCLUDE_DIR)

    if(NOT NUMPY_MULTIARRAY_LIBRARY)
      find_library(NUMPY_MULTIARRAY_LIBRARY multiarray
        HINTS ${NUMPY_PATH}
        PATH_SUFFIXES core)
    endif(NOT NUMPY_MULTIARRAY_LIBRARY)

    if(NOT NUMPY_SCALARMATH_LIBRARY)
      find_library(NUMPY_SCALARMATH_LIBRARY scalarmath
        HINTS ${NUMPY_PATH}
        PATH_SUFFIXES core)
    endif(NOT NUMPY_SCALARMATH_LIBRARY)

    # Reset the library prefix.
    set(CMAKE_FIND_LIBRARY_PREFIXES "${_cmake_find_library_prefixes}")

  endif(PYTHONINTERP_FOUND)
  
  # Handle the QUIETLY and REQUIRED arguments and set NUMPY_FOUND    
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(Numpy DEFAULT_MSG
    NUMPY_MULTIARRAY_LIBRARY NUMPY_SCALARMATH_LIBRARY NUMPY_INCLUDE_DIR)

  # Set non-cached variables
  if(NUMPY_FOUND)
    set(NUMPY_INCLUDE_DIRS "${NUMPY_INCLUDE_DIR}")
    set(NUMPY_LIBRARIES 
      "${NUMPY_MULTIARRAY_LIBRARY}" "${NUMPY_SCALARMATH_LIBRARY}")
  endif(NUMPY_FOUND)
  
  # Find the f2py program
  find_program(F2PY_EXECUTABLE f2py)

endif(NOT NUMPY_FOUND)


## -----------------------------------------------------------------------------
## Macro to generate a Python interface module from one or more Fortran sources
##
## Usage: add_f2py_module(<module-name> <src1>..<srcN> DESTINATION <install-dir>
##
macro (add_f2py_module _name)

  # Precondition check.
  if(NOT F2PY_EXECUTABLE)
    message(FATAL_ERROR "add_f2py_module: f2py executable is not available!")
  endif(NOT F2PY_EXECUTABLE)

  # Parse arguments.
  string(REGEX REPLACE ";?DESTINATION.*" "" _srcs "${ARGN}")
  string(REGEX MATCH "DESTINATION;.*" _dest_dir "${ARGN}")
  string(REGEX REPLACE "^DESTINATION;" "" _dest_dir "${_dest_dir}")

  # Sanity checks.
  if(_srcs MATCHES "^$")
    message(FATAL_ERROR "add_f2py_module: no source files specified")
  endif(_srcs MATCHES "^$")
  if(_dest_dir MATCHES "^$" OR _dest_dir MATCHES ";")
    message(FATAL_ERROR "add_f2py_module: destination directory invalid")
  endif(_dest_dir MATCHES "^$" OR _dest_dir MATCHES ";")

  # Get the compiler-id and map it to compiler vendor as used by f2py.
  # Currently, we only check for GNU, but this can easily be extended. 
  # Cache the result, so that we only need to check once.
  if(NOT F2PY_FCOMPILER)
    if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
      if(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
        set(_fcompiler "gnu95")
      else(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
        set(_fcompiler "gnu")
      endif(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
    else(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
      set(_fcompiler "F2PY_FCOMPILER-NOTFOUND")
    endif(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    set(F2PY_FCOMPILER ${_fcompiler} CACHE STRING
      "F2PY: Fortran compiler type by vendor" FORCE)
    if(NOT F2PY_FCOMPILER)
      message(STATUS "[F2PY]: Could not determine Fortran compiler type. "
                     "Troubles ahead!")
    endif(NOT F2PY_FCOMPILER)
  endif(NOT F2PY_FCOMPILER)

  # Set f2py compiler options: compiler vendor and path to Fortran77/90 compiler.
  if(F2PY_FCOMPILER)
    set(_fcompiler_opts "--fcompiler=${F2PY_FCOMPILER}")
    list(APPEND _fcompiler_opts "--f77exec=${CMAKE_Fortran_COMPILER}")
    if(APPLE)
      list(APPEND _fcompiler_opts "--f77flags=\"-m64\"")
    endif(APPLE)
    if(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
      list(APPEND _fcompiler_opts "--f90exec=${CMAKE_Fortran_COMPILER}")
      if(APPLE)
        list(APPEND _fcompiler_opts "--f90flags=\"-m64\"")
      endif(APPLE)      
    endif(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
  endif(F2PY_FCOMPILER)

  # Make the source filenames absolute.
  set(_abs_srcs)
  foreach(_src ${_srcs})
    get_filename_component(_abs_src ${_src} ABSOLUTE)
    list(APPEND _abs_srcs ${_abs_src})
  endforeach(_src ${_srcs})

  # Get a list of the include directories.
  # The f2py --include_paths option, used when generating a signature file,
  # needs a colon-separated list. The f2py -I option, used when compiling
  # the sources, must be repeated for every include directory.
  get_directory_property(_inc_dirs INCLUDE_DIRECTORIES)
  string(REPLACE ";" ":" _inc_paths "${_inc_dirs}")
  set(_inc_opts)
  foreach(_dir ${_inc_dirs})
    list(APPEND _inc_opts "-I${_dir}")
  endforeach(_dir)

  # Define the command to generate the Fortran to Python interface module. The
  # output will be a shared library that can be imported by python.
  add_custom_command(OUTPUT ${_name}.so
    COMMAND ${F2PY_EXECUTABLE} --quiet -m ${_name} -h ${_name}.pyf
            --include_paths ${_inc_paths} --overwrite-signature ${_abs_srcs}
    COMMAND ${F2PY_EXECUTABLE} --quiet -m ${_name} -c ${_name}.pyf
            ${_fcompiler_opts} ${_inc_opts} ${_abs_srcs}
    DEPENDS ${_srcs}
    COMMENT "[F2PY] Building Fortran to Python interface module ${_name}")

  # Add a custom target <name> to trigger the generation of the python module.
  add_custom_target(${_name} ALL DEPENDS ${_name}.so)

  # Install the python module
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${_name}.so
    DESTINATION ${_dest_dir})

endmacro (add_f2py_module)
