# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(test-idg-api.x)

set(${PROJECT_NAME}_sources main.cpp gridder-common.cpp tGridder.cpp
                            tDegridder.cpp)

# Add boost dynamic link flag for all test files.
# https://www.boost.org/doc/libs/1_66_0/libs/test/doc/html/boost_test/usage_variants.html
# Without this flag, linking is incorrect and boost performs duplicate delete()
# calls after running all tests, in the cleanup phase.
set_source_files_properties(
  ${${PROJECT_NAME}_sources} PROPERTIES COMPILE_DEFINITIONS
                                        "BOOST_TEST_DYN_LINK")

# Set build target
add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_sources})

# link
link_directories(${LIBRARY_OUTPUT_PATH})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..
                    ${CMAKE_CURRENT_SOURCE_DIR}/../.. ${Boost_INCLUDE_DIRS})

set(${PROJECT_NAME}_link_libraries idg-api idg-common idg-external
                                   ${Boost_LIBRARIES})

target_link_libraries(${PROJECT_NAME} ${${PROJECT_NAME}_link_libraries})

install(
  TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION bin/tests/cxx
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static)

add_built_test(${PROJECT_NAME} LABEL unit)
