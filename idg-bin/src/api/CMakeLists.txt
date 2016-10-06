project(idg-api)

# Hack
if (BUILD_LIB_OPENCL)
    include_directories(
        ${CLFFT_INCLUDE_DIRS}
    )
endif()

# sources and header files
set (${PROJECT_NAME}_headers
    Scheme.h
    GridderPlan.h
    DegridderPlan.h
    Datatypes.h
)

set (${PROJECT_NAME}_sources
    Scheme.cpp
    GridderPlan.cpp
    DegridderPlan.cpp
)

add_library(
  ${PROJECT_NAME} OBJECT ${${PROJECT_NAME}_headers}
                         ${${PROJECT_NAME}_sources}
)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-fPIC")

# Copy header files
install(
    FILES
    ${${PROJECT_NAME}_headers}
    DESTINATION
    include/api
)
