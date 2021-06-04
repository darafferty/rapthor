# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

# Create list of files to be installed. This depends on build options.
set(_py_files __init__.py idgtypes.py Proxy.py fft.py Plan.py)

if(BUILD_LIB_CPU)
  list(APPEND _py_files CPU.py)
endif()
if(BUILD_LIB_CUDA)
  list(APPEND _py_files CUDA.py)
endif()
if(BUILD_LIB_CPU AND BUILD_LIB_CUDA)
  list(APPEND _py_files HybridCUDA.py)
endif()
if(BUILD_LIB_OPENCL)
  list(APPEND _py_files OpenCL.py)
endif()

# Install Python modules.
install(
  FILES ${_py_files}
  COMPONENT python
  DESTINATION ${PYTHON_INSTALL_DIR})

# Python-only proxies
add_subdirectory(proxies)
