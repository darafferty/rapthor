# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(utility-initialize)

# external header files needed
include_directories(${CMAKE_SOURCE_DIR}/src/external)

# sources and header files
set(${PROJECT_NAME}_headers Data.h init.h initc.h)

set(${PROJECT_NAME}_sources Data.cpp init.cpp)

# create a common library
add_library(${PROJECT_NAME} OBJECT ${${PROJECT_NAME}_headers}
                                   ${${PROJECT_NAME}_sources})
target_link_libraries(${PROJECT_NAME} PRIVATE xtensor)

# Copy header files
install(FILES ${${PROJECT_NAME}_headers} DESTINATION include/initialize)
