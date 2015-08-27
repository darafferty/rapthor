project(examples-hybrid)

# add subfolders
if (USING_INTEL_CXX_COMPILER)
    add_subdirectory(KNCOffload)
endif()
