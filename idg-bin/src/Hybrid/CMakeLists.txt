project(hybrid)

# add subfolders
add_subdirectory(common)
if (BUILD_WITH_MIC)
   if (USING_INTEL_CXX_COMPILER)
       add_subdirectory(KNC)
   else (USING_INTEL_CXX_COMPILER)
        message("Need Intel compiler to build for Xeon Phi")
   endif(USING_INTEL_CXX_COMPILER)
endif(BUILD_WITH_MIC) 

if (BUILD_WITH_CPU AND BUILD_WITH_CUDA)
   add_subdirectory(CUDA)
endif()
