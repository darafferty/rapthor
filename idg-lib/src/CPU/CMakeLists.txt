project(idg-cpu)

# add subfolders
add_subdirectory(reference)

# currently, only build HaswellEP code 
# with the intel compiler 
if(USING_INTEL_CXX_COMPILER)
add_subdirectory(HaswellEP)
endif(USING_INTEL_CXX_COMPILER)

