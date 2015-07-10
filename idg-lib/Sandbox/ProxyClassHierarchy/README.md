# Implementation of a class hierarchy for the Proxy class

## Questions

* Do we support less numbers of .so files?   
* Where to allocate subgrid? 
* Create the data interfaces: Grid grid(idg::noalloc, ...), grid.dim(),
* grid.size(), grid.layout(), grid.data(), ...      
* Rename chunksize, so that more descriptive? 
* What unit is "field of view"? Arcsec? What is maximal value?
* w_planes -> nr_w_planes?


## FFTW3 linking problem

* export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libfftw3f.so
* Should we link to /usr/lib/x86_64-linux-gnu/libfftw3f_omp.so?
