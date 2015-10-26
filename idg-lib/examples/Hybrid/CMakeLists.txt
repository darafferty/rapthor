project(examples-hybrid)

# add subfolders
if (BUILD_WITH_MIC)
   if (USING_INTEL_CXX_COMPILER)
       add_subdirectory(KNC)
   endif(USING_INTEL_CXX_COMPILER)
endif(BUILD_WITH_MIC) 
