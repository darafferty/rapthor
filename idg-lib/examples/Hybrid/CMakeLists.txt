project(examples-hybrid)

# add subfolders
if (BUILD_LIB_KNC)
   if (USING_INTEL_CXX_COMPILER)
       add_subdirectory(KNC)
   endif(USING_INTEL_CXX_COMPILER)
endif(BUILD_LIB_KNC)

if (BUILD_LIB_CPU AND BUILD_LIB_CUDA)
   add_subdirectory(CUDA)
endif()
