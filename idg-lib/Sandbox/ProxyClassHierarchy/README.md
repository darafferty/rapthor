# Implementation of a class hierarchy for the Proxy class

## Questions

* Parmeters will change to contain observation plus imaging parameters    
* AlgorithmicParameters will contain multptile job sizes, chunk size,   
* etc. with defaults provided by the proxy instance   
* Libraries load at construction and made availbale ...    
* Do we support less numbers of .so files?   
* Where to allocate subgrid? In proxy constructor?
* Create the data interfaces: Grid grid(idg::noalloc, ...), grid.dim(),
* grid.size(), grid.layout(), grid.data(), ...      
* Rename chunksize, so that more descriptive? Same for job_size.
* Name: "ImagingParameters" vs. "AlgorithmicParameters" vs. ...
* What unit is "field of view"? Arcsec? What is maximal value?
* w_planes -> nr_w_planes?
