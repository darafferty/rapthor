
#include <iostream>
#include <iomanip>
#include <sstream>
#include "AlgorithmParameters.h" 

using namespace std;


namespace idg {

  // set methods
  void AlgorithmParameters::set_subgrid_size(unsigned int sgs)
  {
    subgrid_size = sgs;
  }
  
  void AlgorithmParameters::set_chunk_size(unsigned int cs) 
  {
    if (cs > 0) chunk_size = cs; // avoid chunk_size==0 
  }
    
  void AlgorithmParameters::set_job_size(unsigned int js)
  {
    if (js == 0) return; // avoid job_size==0
    job_size = js;
    job_size_gridding = js;
    job_size_degridding = js;
    job_size_gridder = js; 
    job_size_adder = js; 
    job_size_splitter = js; 
    job_size_degridder = js; 
  }

  void AlgorithmParameters::set_job_size_gridding(unsigned int js)
  {
    if (js == 0) return; // avoid job_size==0
    job_size_gridding = js;
    job_size_gridder = js; 
    job_size_adder = js; 
  }
  
  void AlgorithmParameters::set_job_size_degridding(unsigned int js)
  {
    if (js == 0) return; // avoid job_size==0
    job_size_degridding = js;
    job_size_splitter = js; 
    job_size_degridder = js; 
  }

  void AlgorithmParameters::set_job_size_gridder(unsigned int js)
  {
    if(js > 0) job_size_gridder = js;
  }
  
  void AlgorithmParameters::set_job_size_adder(unsigned int js)
  {
    if(js > 0) job_size_adder = js;
  }
  
  void AlgorithmParameters::set_job_size_splitter(unsigned int js)
  {
    if(js > 0) job_size_splitter = js;
  } 

  void AlgorithmParameters::set_job_size_degridder(unsigned int js)
  {
    if(js > 0) job_size_degridder = js;
  }


  // auxiliary functions
  void AlgorithmParameters::print(ostream& os) const
  {
    const int fw1 = 30;
    const int fw2 = 10;

    os << "ALGORITHM PARAMETERS:" << endl;
    
    os << setw(fw1) << left << "Subgrid size" << "== " 
       << setw(fw2) << right << subgrid_size << endl;
    
    os << setw(fw1) << left << "Chunk size" << "== " 
       << setw(fw2) << right << chunk_size << endl;
        
    os << setw(fw1) << left << "Job size" << "== " 
       << setw(fw2) << right << job_size << endl;

    os << setw(fw1) << left << "Job size (gridding)" << "== " 
       << setw(fw2) << right << job_size_gridding << endl;

    os << setw(fw1) << left << "Job size (degridding)" << "== " 
       << setw(fw2) << right << job_size_degridding << endl;

    os << setw(fw1) << left << "Job size (gridder)" << "== " 
       << setw(fw2) << right << job_size_gridder << endl;

    os << setw(fw1) << left << "Job size (adder)" << "== " 
       << setw(fw2) << right << job_size_adder << endl;

    os << setw(fw1) << left << "Job size (splitter)" << "== " 
       << setw(fw2) << right << job_size_splitter << endl;

    os << setw(fw1) << left << "Job size (degridder)" << "== " 
       << setw(fw2) << right << job_size_degridder << endl;
  }


  void AlgorithmParameters::print() const
  {
    print(cout);
  }


  void AlgorithmParameters::read_parameters_from_env() 
  {
    const unsigned int DEFAULT_SUBGRIDSIZE = 0;
    const unsigned int DEFAULT_CHUNKSIZE = 1;
    const unsigned int DEFAULT_JOBSIZE = 1;

    // subgrid_size
    char *cstr_subgrid_size = getenv(ENV_SUBGRIDSIZE.c_str());
    if (cstr_subgrid_size != nullptr) {
      subgrid_size = atoi(cstr_subgrid_size);
    } else {
      subgrid_size = DEFAULT_SUBGRIDSIZE;
    }

    // chunk_size
    char *cstr_chunk_size = getenv(ENV_CHUNKSIZE.c_str());
    if (cstr_chunk_size != nullptr) {
      chunk_size = atoi(cstr_chunk_size);
      if (chunk_size < 1)
	chunk_size = DEFAULT_CHUNKSIZE;
    } else {
      chunk_size = DEFAULT_CHUNKSIZE;
    }

    // job_size
    char *cstr_job_size = getenv(ENV_JOBSIZE.c_str());
    if (cstr_job_size != nullptr) {
      set_job_size( atoi(cstr_job_size) );
      if (job_size < 1)
	set_job_size( DEFAULT_JOBSIZE );
    } else {
      set_job_size( DEFAULT_JOBSIZE );
    }

    // so far finer control over jobsizes not via environment 
    // must use set methods

  } // read_parameters_from_env()


  string AlgorithmParameters::definitions(unsigned int subgrid_size,
					  unsigned int chunk_size,
					  unsigned int job_size)
  {
    stringstream parameters;
    parameters << " -DSUBGRIDSIZE=" << subgrid_size;
    parameters << " -DCHUNKSIZE=" << chunk_size;
    return parameters.str();
  }


  // helper functions
  ostream& operator<<(ostream& os, const AlgorithmParameters& ap) 
  {
    ap.print(os);
    return os;
  }


} // namespace idg
