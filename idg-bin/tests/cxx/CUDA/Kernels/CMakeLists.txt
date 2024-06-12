add_library(TestFixture OBJECT TestFixture.cpp)
target_link_libraries(TestFixture ${LINK_LIBRARIES})

set(LINK_LIBRARIES ${LINK_LIBRARIES} Boost::unit_test_framework TestFixture)

set(GRIDDER test-cuda-kernel-gridder.x)
add_executable(${GRIDDER} gridder.cpp)
target_link_libraries(${GRIDDER} ${LINK_LIBRARIES})

set(DEGRIDDER test-cuda-kernel-degridder.x)
add_executable(${DEGRIDDER} degridder.cpp)
target_link_libraries(${DEGRIDDER} ${LINK_LIBRARIES})

set(FFT test-cuda-kernel-fft.x)
add_executable(${FFT} fft.cpp)
target_link_libraries(${FFT} ${LINK_LIBRARIES})

add_built_test(${GRIDDER} LABEL bin_cxx)
add_built_test(${DEGRIDDER} LABEL bin_cxx)
add_built_test(${FFT} LABEL bin_cxx)
