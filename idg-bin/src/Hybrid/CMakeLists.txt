project(hybrid)

# add subfolders
if (BUILD_LIB_KNC)
   if (INTEL_CXX_COMPILER)
       add_subdirectory(KNC)
   else (INTEL_CXX_COMPILER)
        message("Need Intel compiler to build for Xeon Phi")
   endif(INTEL_CXX_COMPILER)
endif(BUILD_LIB_KNC)

if (BUILD_LIB_CPU AND BUILD_LIB_CUDA)
    add_subdirectory(CUDA)
endif()
