# Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(cpu-optimized-mp.x)

# Set sources
set (${PROJECT_NAME}_sources 
    main.cpp
)

# Set build target
add_executable (${PROJECT_NAME} ${${PROJECT_NAME}_sources})

# Set include directories
target_include_directories(
    ${PROJECT_NAME} PRIVATE
    ${MPI_C_COMPILER_INCLUDE_DIRS}
)

# Set link libaries
set(LINK_LIBRARIES
    idg-util
    idg-cpu
    ${MPI_mpi_LIBRARY}
    ${MPI_mpi_cxx_LIBRARY}
)

target_link_libraries (
    ${PROJECT_NAME}
    ${LINK_LIBRARIES}
)

# install
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin/examples/cxx
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static
)
