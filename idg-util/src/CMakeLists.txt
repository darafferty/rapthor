add_subdirectory(initialize)
add_subdirectory(external)

#########################################################
# build libidg-util: init example data, ...
#########################################################
add_library(
    idg-util
    SHARED
    $<TARGET_OBJECTS:utility-initialize>
    $<TARGET_OBJECTS:utility-external>
)

set_target_properties(
    idg-util
    PROPERTIES
    PUBLIC_HEADER idg-util.h)

install(
    TARGETS idg-util
    EXPORT IDGTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static
    PUBLIC_HEADER DESTINATION include
)
