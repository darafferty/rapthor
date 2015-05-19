# Install the header file need at compile time
install(FILES Proxies.h DESTINATION include/idg/XEON)

# Install files needed for runtime compilation
install(
  FILES Wrapper.cpp Kernels.h Kernels.cpp Gridder.cpp Degridder.cpp FFT.cpp Adder.cpp Splitter.cpp Shifter.cpp
  DESTINATION lib/idg/XEON)
  

