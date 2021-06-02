# Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(test-hybrid-cuda-unified-optimized.x)

set(${PROJECT_NAME}_sources main.cpp ${IDG_BIN_TEST_CXX_COMMON_DIR}/common.cpp)

add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_sources})

target_link_libraries(${PROJECT_NAME} ${LINK_LIBRARIES})

install(
  TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION bin/tests/cxx
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static)
