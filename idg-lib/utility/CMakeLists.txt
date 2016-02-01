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

set_target_properties(idg-utility PROPERTIES
                      PUBLIC_HEADER idg-utility.h)

install(TARGETS idg-utility
  EXPORT IDGTargets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
  PUBLIC_HEADER DESTINATION include/utility
)
