set(PYTHON_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/lib/python2.7/dist-packages)

install(FILES
        util.py
        data.py
        DESTINATION
        ${PYTHON_INSTALL_DIR})
