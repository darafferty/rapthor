project(idg-hybrid)

# add subfolders
if (BUILD_MIC_OFFLOAD)
   if (USING_INTEL_CXX_COMPILER)
       add_subdirectory(KNC)
   else (USING_INTEL_CXX_COMPILER)
        message("Need Intel compiler to build MIC offload code.")
   endif(USING_INTEL_CXX_COMPILER)
endif(BUILD_MIC_OFFLOAD) 
