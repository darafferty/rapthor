# Install Python examples
install(FILES
  cpu-reference.py
  cpu-sandybridgeep.py
  cpu-haswellep.py
  hybrid-maxwellhaswellep.py
  hybrid-knc.py
  demo-idg.py
  cpu-clean-reference.py
  cpu-image-haswellep.py
  opencl-reference.py
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  DESTINATION bin/examples)
