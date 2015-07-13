#include <iostream>
#include <cstdint>
#include <omp.h>

using namespace std;

namespace idg {
    namespace auxiliary {


      void report(const char *name, 
		  double runtime, 
		  uint64_t flops, 
		  uint64_t bytes) 
      {
#pragma omp critical (clog)
	{
	  clog << name << ": " << runtime << " s";
	  if (flops != 0)
	    clog << ", " << flops / runtime * 1e-9 << " GFLOPS";
	  if (bytes != 0)
	    clog << ", " << bytes / runtime * 1e-9 << " GB/s";
	  clog << endl;
	}
      }      

      
      void report_runtime(double runtime) 
      {
	clog << "runtime: " << runtime << " s" << endl;
      }
      
      
      void report_visibilities(double runtime,
			       uint64_t nr_baselines,
			       uint64_t nr_time,
			       uint64_t nr_channels) 
      {
	uint64_t nr_visibilities = nr_baselines * nr_time * nr_channels;
	clog << "throughput: " << 1e-6 * nr_visibilities / runtime 
	     << " Mvisibilities/s" << endl;
      }
      
      void report_subgrids(double runtime,
			   uint64_t nr_baselines) 
      {
	clog << "throughput: " << 1e-3 * nr_baselines / runtime 
	     << " Ksubgrids/s" << endl;
      }

      
    } // namespace auxiliary
} // namespace idg
      
