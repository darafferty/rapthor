cmake_minimum_required(VERSION 2.6)

project(ImageDomainGridder)

include_directories(include)


FIND_PACKAGE( Boost 1.40 COMPONENTS filesystem REQUIRED )

INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )


add_subdirectory(XEON)
add_subdirectory(Common)


# add_executable(IDG-XEON XEON/IDG.cpp XEON/Proxies.cpp Common/RW.cpp Common/Init.cpp Common/Util.cpp)
# target_link_libraries(IDG-XEON dl)

add_library(idg SHARED Common/RW.cpp Common/Init.cpp Common/Util.cpp XEON/Proxies.cpp )
TARGET_LINK_LIBRARIES( idg ${Boost_LIBRARIES} )

install(TARGETS idg DESTINATION lib)


