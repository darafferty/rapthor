project(demo-hdf5.x)

include_directories(
  ${HDF5_CXX_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIR}
)

set (${PROJECT_NAME}_headers
  MS-hdf5.h
)

set (${PROJECT_NAME}_sources
  demo.cpp
  MS-hdf5.cpp
)

# Set build target
add_executable (${PROJECT_NAME}
  ${${PROJECT_NAME}_headers}
  ${${PROJECT_NAME}_sources})

set(LINK_LIRBRARIES
  idg
  ${HDF5_hdf5_cpp_LIBRARY}
  ${HDF5_hdf5_LIBRARY}
  ${OpenCV_LIBS})

target_link_libraries (
  ${PROJECT_NAME}
  ${LINK_LIRBRARIES}
)

# install
install(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION bin/demo/cxx
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
)
