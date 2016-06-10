# Install Python examples
install(FILES
  api-gridder.py
  api-degridder.py
  api-demo.py
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  DESTINATION bin/examples)
