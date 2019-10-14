project(external)

# workaround for compilation warning with gcc/8.3.0 and boost/1.66
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=parentheses")

# external header files
if (Boost_FOUND)
include_directories(${Boost_INCLUDE_DIR})
endif()

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
