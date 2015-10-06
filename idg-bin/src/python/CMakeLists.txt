find_package(Boost REQUIRED COMPONENTS python)
find_package(PythonLibs)

find_package(FFTW3)
find_package(Numpy)

include_directories(
  ${PYTHON_INCLUDE_DIRS}
  ${NUMPY_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/common
)

add_library(_idg MODULE idg.cpp)
TARGET_LINK_LIBRARIES( 
  _idg
  idg-cpu-common
  idg-common
  idg-cpu-haswellep
  idg-cpu-reference
  ${Boost_LIBRARIES} 
#  ${FFTW3_LIBRARY} 
)

set_target_properties(_idg PROPERTIES
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/idg")

set(PYTHON_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/lib/python2.7/dist-packages)
# Install Python modules.
install(FILES __init__.py DESTINATION ${PYTHON_INSTALL_DIR}/idg/)
install(TARGETS _idg DESTINATION ${PYTHON_INSTALL_DIR}/idg/)

install(
  FILES testidg.py 
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)



