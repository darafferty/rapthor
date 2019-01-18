project(fft)

# external header files
include_directories(
    ${FFTW3_INCLUDE_DIR}
)

# sources and header files
set (${PROJECT_NAME}_headers
    FFT.h
    FFTC.h
)

set (${PROJECT_NAME}_sources
    FFT.cpp
)

# create library
add_library(
    ${PROJECT_NAME} OBJECT ${${PROJECT_NAME}_headers}
                           ${${PROJECT_NAME}_sources}
    )
if (NOT BUILD_STATIC_LIBS)
    set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-fPIC" SOVERSION 0)
endif()

# Copy header files
install(
    FILES
    ${${PROJECT_NAME}_headers}
    DESTINATION
    include/fft
)
