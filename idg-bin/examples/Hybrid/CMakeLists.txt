project(examples-hybrid)

# add subfolders
if (BUILD_MIC_OFFLOAD)
   if (USING_INTEL_CXX_COMPILER)
       add_subdirectory(KNC)
   endif(USING_INTEL_CXX_COMPILER)
endif(BUILD_MIC_OFFLOAD) 
