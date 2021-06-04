# Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(cpu-optimized-mp.x)

# Set sources
set(${PROJECT_NAME}_sources main.cpp)

# Set build target
add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_sources})

# Set include directories
target_include_directories(
  ${PROJECT_NAME}
  PRIVATE ${MPI_CXX_INCLUDE_DIRS} ${PROJECT_NAME}
  PRIVATE ${FTI_INCLUDE_DIR})

# Set link libaries
target_link_libraries(${PROJECT_NAME} PRIVATE idg-util idg-cpu)
target_link_libraries(${PROJECT_NAME} PRIVATE ${MPI_mpi_LIBRARY})
target_link_libraries(${PROJECT_NAME} PRIVATE ${MPI_mpi_cxx_LIBRARY})
if(${BUILD_WITH_FTI})
  target_link_libraries(${PROJECT_NAME} PRIVATE fti.shared)
endif()

# install
install(
  TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION bin/examples/cxx
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static)
