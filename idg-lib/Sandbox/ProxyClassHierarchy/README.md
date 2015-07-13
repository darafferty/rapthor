# Implementation of a class hierarchy for the Proxy class


## To do

* Sanity check for paramters: e.g., now if grid_size=128, main.x crashes
* Create the data interfaces: Grid grid(idg::noalloc, ...), grid.dim(),
  grid.size(), grid.layout(), grid.data(), ...      
* Check if loaded .so has the correct paramters (importent if not cerated
  before, but existing loaded)
* put .so in tmp/tmp-XXXXXX by default


## FFTW3 linking problem

* export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libfftw3f.so
* Should we link to /usr/lib/x86_64-linux-gnu/libfftw3f_omp.so?


## Questions

* Rename chunksize, so that more descriptive? 
* What unit is "field of view"? Arcsec? What is maximal value?

