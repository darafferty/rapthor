# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(test-idg-lib.x)

set (${PROJECT_NAME}_sources
    runtests.cpp
    tComputeN.cpp
)

# Add boost dynamic link flag for all test files.
# https://www.boost.org/doc/libs/1_66_0/libs/test/doc/html/boost_test/usage_variants.html
# Without this flag, linking is incorrect and boost performs duplicate delete()
# calls after running all tests, in the cleanup phase.
set_source_files_properties(
    ${${PROJECT_NAME}_sources} PROPERTIES COMPILE_DEFINITIONS "BOOST_TEST_DYN_LINK"
)

# Set build target
add_executable (${PROJECT_NAME} ${${PROJECT_NAME}_sources})

# link
link_directories (${LIBRARY_OUTPUT_PATH})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set (${PROJECT_NAME}_link_libraries
    ${IDG_LIBRARY}
    ${Boost_LIBRARIES}
)

target_link_libraries (
    ${PROJECT_NAME}
    ${${PROJECT_NAME}_link_libraries}
)

# install
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin/tests/cxx
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static
)

# test
add_test(
    NAME ${PROJECT_NAME}
    COMMAND ${PROJECT_NAME})
