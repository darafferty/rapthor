set(PYTHON_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/lib/python2.7/dist-packages)

install(
    FILES
    Python.py
    PythonReference.py
    DESTINATION
    ${PYTHON_INSTALL_DIR})
