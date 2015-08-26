project(idg-examples-external)

# sources and header files
set (${PROJECT_NAME}_headers 
  uvwsim.h
)

set (${PROJECT_NAME}_sources 
  uvwsim.cpp
)

# create a common library 
# add_library(
#    ${PROJECT_NAME} STATIC ${${PROJECT_NAME}_headers}
#                           ${${PROJECT_NAME}_sources}
#)
add_library(
    ${PROJECT_NAME} ${${PROJECT_NAME}_headers}
                    ${${PROJECT_NAME}_sources}
)

# linker
target_link_libraries(${PROJECT_NAME} idg-examples-external)

# installer
# install(TARGETS ${PROJECT_NAME} 
#   RUNTIME DESTINATION bin
#   LIBRARY DESTINATION lib
#   ARCHIVE DESTINATION lib/static
#   )
