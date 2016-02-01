project(utility)

# add subfolders
add_subdirectory(initialize)
add_subdirectory(external)


####################################################
# build libidg-utility, containing all common code #
####################################################
add_library(idg-utility
  $<TARGET_OBJECTS:utility-initialize>
  $<TARGET_OBJECTS:utility-external>
)

install(TARGETS idg-utility
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
)

install(
    FILES
    idg-utility.h
    DESTINATION
    include/utility
)
