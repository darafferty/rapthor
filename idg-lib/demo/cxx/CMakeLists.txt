project(demo-cxx)

# if found casacore
find_package(CasaCore)
if (CASACORE_FOUND)
add_subdirectory(Casacore)
endif()

# if found HDF5
find_package(HDF5 COMPONENTS CXX)
if (HDF5_FOUND)
add_subdirectory(HDF5)
endif()
