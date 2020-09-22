# List of integration tests
set(INTEGRATION_TESTS
  tSinglePointSource
  tGridDegrid
)

# Add integration tests to "integration", can be invoked with 
# "ctest -L integration"
foreach(TEST ${INTEGRATION_TESTS})
  add_test(${TEST} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST}.sh)
  set_tests_properties(${TEST} PROPERTIES LABELS "integration")
endforeach()