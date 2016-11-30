# Install Python tests
install(FILES
    test-hybrid-cuda.py
    test-degridder-001.py
    test-degridder-002.py
    test-cuda-generic.py
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    DESTINATION bin/tests/python)
