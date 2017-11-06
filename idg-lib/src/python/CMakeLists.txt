set(PYTHON_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/lib/python2.7/dist-packages)

# Install Python modules.
install(FILES
        idg.py
        idgtypes.py
        CPU.py
        CUDA.py
        OpenCL.py
        HybridCUDA.py
        Proxy.py
        fft.py
        Plan.py
        DESTINATION
        ${PYTHON_INSTALL_DIR})

# Python-only proxies
add_subdirectory(proxies)
