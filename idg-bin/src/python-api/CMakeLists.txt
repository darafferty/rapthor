set(PYTHON_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/lib/python2.7/dist-packages)

# Install Python modules.
install(FILES
        IDG.py
        DESTINATION
        ${PYTHON_INSTALL_DIR})
