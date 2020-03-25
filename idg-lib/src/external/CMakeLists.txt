project(external)

# sources and header files
set (${PROJECT_NAME}_headers
    ThrowAssert.hpp
    Stopwatch.h
)

set (${PROJECT_NAME}_sources
    Stopwatch.cpp
)

add_library(
	${PROJECT_NAME} OBJECT ${${PROJECT_NAME}_headers}
	                       ${${PROJECT_NAME}_sources}
)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-fPIC" SOVERSION 0)

# Copy header files
install(
	FILES
	${${PROJECT_NAME}_headers}
	DESTINATION
	include/external
)
