# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

#project(test-cuda-wtiling)

# Forward test
set(FORWARD test-cuda-wtiling-forward.x)
add_executable(${FORWARD} forward.cpp)

target_link_libraries(${FORWARD} ${LINK_LIBRARIES})

target_include_directories(${FORWARD} PRIVATE ${CUDA_INCLUDE_DIR})

# Backward test
set(BACKWARD test-cuda-wtiling-backward.x)
add_executable(${BACKWARD} backward.cpp)

target_link_libraries(${BACKWARD} ${LINK_LIBRARIES})

target_include_directories(${BACKWARD} PRIVATE ${CUDA_INCLUDE_DIR})

# Set tests
add_built_test(${FORWARD} LABEL bin_cxx)
add_built_test(${BACKWARD} LABEL bin_cxx)
