# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(utility-external)

# sources and header files
set (${PROJECT_NAME}_headers 
  uvwsim.h
)

set (${PROJECT_NAME}_sources 
  uvwsim.cpp
)

# create a common library 
add_library(
    ${PROJECT_NAME} OBJECT ${${PROJECT_NAME}_headers}
                           ${${PROJECT_NAME}_sources}
)

# Copy header files
install(
    FILES
    ${${PROJECT_NAME}_headers}
    DESTINATION
    include/external
)
