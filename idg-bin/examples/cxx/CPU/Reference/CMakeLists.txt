project(cpu-reference.x)

# Set sources
set (${PROJECT_NAME}_sources
    main.cpp
)

# Set build target
add_executable (${PROJECT_NAME} ${${PROJECT_NAME}_sources})

target_link_libraries (
    ${PROJECT_NAME} 
    ${LINK_LIBRARIES}
)

# install
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin/examples/cxx
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static
)
