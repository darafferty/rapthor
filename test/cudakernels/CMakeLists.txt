add_test(NAME tCudaGridder
         COMMAND python3 -m pytest -v --exitfirst
                 "${CMAKE_CURRENT_SOURCE_DIR}/test-cuda-gridder.py")

add_test(NAME tCudaDegridder
         COMMAND python3 -m pytest -v --exitfirst
                 "${CMAKE_CURRENT_SOURCE_DIR}/test-cuda-gridder.py")
