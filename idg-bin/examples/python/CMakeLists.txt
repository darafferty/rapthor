install(FILES
    common.py
    cpu-reference.py
    cpu-optimized.py
    cuda-generic.py
    cuda-unified.py
    opencl-generic.py
    hybrid-cuda-generic-optimized.py
    coverage.py
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    DESTINATION bin/examples/python)
