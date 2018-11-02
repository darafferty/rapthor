add_subdirectory(external)
include_directories(external)

add_subdirectory(initialize)

if (BUILD_WITH_PYTHON)
    add_subdirectory(python)
endif()

#########################################################
# build libidg-util: init example data, ...
#########################################################
add_library(
    idg-util
    SHARED
    $<TARGET_OBJECTS:utility-initialize>
    $<TARGET_OBJECTS:utility-external>
)

set(LINK_LIBRARIES idg-common idg-fft)

target_link_libraries(
    idg-util
    ${LINK_LIBRARIES}
)

set_target_properties(
    idg-util
    PROPERTIES
    PUBLIC_HEADER idg-util.h)

install(
    TARGETS idg-util
    EXPORT  idg-util-targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static
    PUBLIC_HEADER DESTINATION include
)
