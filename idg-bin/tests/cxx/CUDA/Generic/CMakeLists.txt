project(test-cuda-generic.x)

# Set sources
set (
    ${PROJECT_NAME}_sources
    main.cpp
)

# Set build target
add_executable (
    ${PROJECT_NAME} ${${PROJECT_NAME}_sources}
)

target_link_libraries (
    ${PROJECT_NAME}
    ${LINK_LIBRARIES}
)

# Set install
install (
    TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin/tests/cxx
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static
)

# Set test
add_test(
    NAME ${PROJECT_NAME}
    COMMAND ${CMAKE_INSTALL_PREFIX}/bin/tests/cxx/${PROJECT_NAME})
