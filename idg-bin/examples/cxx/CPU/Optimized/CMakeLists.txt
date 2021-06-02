# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(cpu-optimized.x)

# Set sources
set(${PROJECT_NAME}_sources main.cpp)

# Set build target
add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_sources})

target_link_libraries(${PROJECT_NAME} ${LINK_LIBRARIES})

# install
install(
  TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION bin/examples/cxx
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static)
