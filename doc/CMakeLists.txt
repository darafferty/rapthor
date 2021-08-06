find_package(Sphinx)
find_package(Doxygen)

if(NOT (SPHINX_FOUND AND DOXYGEN_FOUND))
  set(missing_packages "Doxygen and Sphinx")
  if(SPHINX_FOUND)
    set(missing_packages "Doxygen")
  endif()
  if(DOXYGEN_FOUND)
    set(missing_packages "Sphinx")
  endif()
  add_custom_target(
    doc
    COMMAND
      echo -e
      "\\nTo make the documentation please install ${missing_packages}, rerun cmake and then run make doc\\n"
    COMMAND exit 1
    VERBATIM)
else()

  find_package(Doxygen REQUIRED)

  # Find all the public headers
  # get_target_property(CAT_CUTIFIER_PUBLIC_HEADER_DIR CatCutifier INTERFACE_INCLUDE_DIRECTORIES)
  # file(GLOB_RECURSE CAT_CUTIFIER_PUBLIC_HEADERS ${CAT_CUTIFIER_PUBLIC_HEADER_DIR}/*.h)

  set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/doxygen")
  file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})

  set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
  set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

  #Replace variables inside @@ with the current values
  configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

  #This will be the main output of our command
  set(DOXYGEN_INDEX_FILE "${DOXYGEN_OUTPUT_DIR}/xml/index.xml")

  add_custom_command(
    OUTPUT ${DOXYGEN_INDEX_FILE}
    # DEPENDS ${CAT_CUTIFIER_PUBLIC_HEADERS}
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
    WORKING_DIRECTORY ${DOXYGEN_OUTPUT_DIR}
    MAIN_DEPENDENCY Doxyfile
    COMMENT "Generating doxygen docs")

  set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
  set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/html)

  add_custom_target(
    doc
    COMMAND
      ${SPHINX_EXECUTABLE} -b html ${SPHINX_SOURCE} ${SPHINX_BUILD}
      # Tell Breathe where to find the Doxygen output
      -Dbreathe_projects.IDG=${DOXYGEN_OUTPUT_DIR}/xml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating documentation with Sphinx"
    DEPENDS ${DOXYGEN_INDEX_FILE})

endif()
