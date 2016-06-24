project(demo.x)

include_directories(
  ${CASACORE_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/utility   # TODO: remove
)

set (${PROJECT_NAME}_sources
  demo.cpp
)

# Set build target
add_executable (${PROJECT_NAME} ${${PROJECT_NAME}_sources})

set(LINK_LIRBRARIES
  idg
  idg-utility    # TODO: remove
  ${CASACORE_LIBRARIES}
  ${OpenCV_HIGHGUI_LIBRARY}
  ${OpenCV_CORE_LIBRARY})

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
