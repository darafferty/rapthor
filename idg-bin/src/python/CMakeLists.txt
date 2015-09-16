find_package(Boost REQUIRED COMPONENTS python)
find_package(PythonLibs)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
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
  ${Boost_LIBRARIES} 
)

set_target_properties(_idg PROPERTIES
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/idg")

add_custom_target( init ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py ${CMAKE_BINARY_DIR}/lib/idg
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/testidg.py ${CMAKE_BINARY_DIR}/lib)
  
# Install Python modules.
# install(FILES __init__.py DESTINATION ${PYTHON_INSTALL_DIR}/idg)


