set(PYTHON_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/lib/python2.7/dist-packages/idg)

# Install Python modules.
install(FILES
        __init__.py
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
