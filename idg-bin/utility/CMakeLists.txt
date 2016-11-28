project(utility)

# add subfolders
add_subdirectory(data)
add_subdirectory(initialize)
add_subdirectory(external)

#########################################################
# build libidg-utility: init example data, ...
#########################################################
add_library(idg-utility
    $<TARGET_OBJECTS:utility-initialize>
    $<TARGET_OBJECTS:utility-external>
)

target_link_libraries(
    idg-utility
    idg-fft
)

set_target_properties(idg-utility PROPERTIES
                      PUBLIC_HEADER idg-utility.h)

install(TARGETS idg-utility
    EXPORT IDGTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static
    PUBLIC_HEADER DESTINATION include
)
