project(test-idg-api.x)


set (${PROJECT_NAME}_sources
    main.cpp
)

# Set build target
add_executable (${PROJECT_NAME} ${${PROJECT_NAME}_sources})

# link
link_directories (${LIBRARY_OUTPUT_PATH})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set (${PROJECT_NAME}_link_libraries
    ${IDG_LIBRARY}
    idg-api
    ${LAPACKE_LIB}
)

target_link_libraries (
    ${PROJECT_NAME}
    ${${PROJECT_NAME}_link_libraries}
)

# install
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin/tests/cxx
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static
)

# test
add_test(
    NAME ${PROJECT_NAME}
    COMMAND ${CMAKE_INSTALL_PREFIX}/bin/tests/cxx/${PROJECT_NAME})
