# Implementation of a class hierarchy for the Proxy class

## Questions

* Should we have 
  Proxy::CPU xeon(Compiler c, CompilerFlags flags, ...)
  such that the provided source is in ONE language only or
  Proxy::CPU xeon(CompilerEnvironment cc, ...)
  and we can make use of everything that is set in cc?
* Rename chunksize, so that more descriptive? Same for job_size.
* Name: "ImagingParameters" vs. "AlgorithmicParameters" vs. ...
* What unit is "field of view"? Arcsec? What is maximal value?
* w_planes -> nr_w_planes?
