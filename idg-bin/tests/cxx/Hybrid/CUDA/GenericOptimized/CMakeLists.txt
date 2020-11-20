# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(test-hybrid-cuda-generic-optimized.x)

# Set sources
set (
    ${PROJECT_NAME}_sources
    main.cpp
)

# Set build target
add_executable (
    ${PROJECT_NAME} ${${PROJECT_NAME}_sources}
)

# Set libraries to link
link_directories (${LIBRARY_OUTPUT_PATH})

target_link_libraries (
    ${PROJECT_NAME}
    ${LINK_LIBRARIES}
)

# Set install
install (
    TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin/tests/cxx
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static
)

# Set test
add_test(
    NAME ${PROJECT_NAME}
    COMMAND ${CMAKE_INSTALL_PREFIX}/bin/tests/cxx/${PROJECT_NAME})
