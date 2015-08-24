project(idg-common)

# sources and header files
set (${PROJECT_NAME}_headers 
  AbstractProxy.h
  CompilerEnvironment.h
  Parameters.h
  ProxyInfo.h
  RuntimeWrapper.h
  auxiliary.h
)

set (${PROJECT_NAME}_sources 
  CompilerEnvironment.cpp 
  Parameters.cpp 
  ProxyInfo.cpp 
  RuntimeWrapper.cpp 
  auxiliary.cpp 
)

# create a common library 
add_library(
    ${PROJECT_NAME} SHARED ${${PROJECT_NAME}_headers}
                           ${${PROJECT_NAME}_sources}
)

# linker
target_link_libraries(${PROJECT_NAME} idg-common)

# installer
install(TARGETS ${PROJECT_NAME} 
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
)