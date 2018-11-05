install(FILES
    common.py
    test-cpu-optimized.py
    test-cuda-generic.py
    test-hybrid-cuda-generic-optimized.py
    test-opencl-generic.py
    test-python-reference.py
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    DESTINATION bin/tests/python)
