# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

project(fft)

# sources and header files
set(${PROJECT_NAME}_headers FFT.h FFTC.h)

set(${PROJECT_NAME}_sources FFT.cpp)

# Compiler options
add_compile_options("-O3")

# create library
add_library(${PROJECT_NAME} OBJECT ${${PROJECT_NAME}_headers}
                                   ${${PROJECT_NAME}_sources})
target_link_libraries(${PROJECT_NAME} PRIVATE PkgConfig::FFTW)

# Copy header files
install(FILES ${${PROJECT_NAME}_headers} DESTINATION include/fft)
