# Install Python tests
install(FILES
    test-cpu-reference.py
    test-cpu-haswellep.py
    test-hybrid-maxwellhaswellep.py
    test-degridder-001.py
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    DESTINATION bin/tests)
