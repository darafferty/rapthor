# Install Python examples
install(FILES
  demo-idg.py
  demo-idg2.py
  demo-hdf5.py
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  DESTINATION bin/demo/python)
