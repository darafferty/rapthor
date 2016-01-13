project(hybrid)

# add subfolders
if (BUILD_LIB_KNC)
   if (USING_INTEL_CXX_COMPILER)
       add_subdirectory(KNC)
   else (USING_INTEL_CXX_COMPILER)
        message("Need Intel compiler to build for Xeon Phi")
   endif(USING_INTEL_CXX_COMPILER)
endif(BUILD_LIB_KNC)

if (BUILD_LIB_CPU AND BUILD_LIB_CUDA)
    message("Hybrid CUDA is disabled")
    #add_subdirectory(CUDA)
endif()
