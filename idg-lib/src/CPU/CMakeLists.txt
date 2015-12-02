project(idg-cpu)

# add subfolders
add_subdirectory(common)
add_subdirectory(Reference)
add_subdirectory(SandyBridgeEP)
add_subdirectory(HaswellEP)

add_library(${PROJECT_NAME}
  $<TARGET_OBJECTS:idg-cpu-common>
  $<TARGET_OBJECTS:idg-cpu-reference>
  $<TARGET_OBJECTS:idg-cpu-haswellep>
  $<TARGET_OBJECTS:idg-cpu-sandybridgeep>
  )

install(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  )
