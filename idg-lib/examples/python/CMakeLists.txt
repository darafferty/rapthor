# Install Python examples
install(FILES
  common.py
  cpu-reference.py
  cpu-optimized.py
  hybrid-cuda.py
  hybrid-knc.py
  hybrid-opencl.py
  cpu-clean-reference.py
  cpu-image-optimized.py
  cuda-generic.py
  opencl-generic.py
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  DESTINATION bin/examples/python)
