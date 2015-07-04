
#include <iostream>
#include <iomanip>
#include "AlgorithmicParameters.h" 

using namespace std;


namespace idg {

  // set methods
  void AlgorithmicParameters::set_grid_size(unsigned int gs)
  {
    grid_size = gs;
  }

  void AlgorithmicParameters::set_subgrid_size(unsigned int sgs)
  {
    subgrid_size = sgs;
  }
  
  void AlgorithmicParameters::set_chunk_size(unsigned int cs) 
  {
    chunk_size = cs;
  }
    
  void AlgorithmicParameters::set_w_planes(unsigned int wp) 
  { 
    cerr << "Setting the number of polarizations is currently not supported." << endl;    
  }

  void AlgorithmicParameters::set_job_size(unsigned int js)
  {
    job_size = js;
  }


  // auxiliary functions
  void AlgorithmicParameters::print(ostream& os) const
  {
    const int fw1 = 30;
    const int fw2 = 10;

    os << "ALGORITHMIC PARAMETERS:" << endl;
    
    os << setw(fw1) << left << "Grid size" << "== " 
       << setw(fw2) << right << grid_size << endl;
    
    os << setw(fw1) << left << "Subgrid size" << "== " 
       << setw(fw2) << right << subgrid_size << endl;
    
    os << setw(fw1) << left << "Chunk size" << "== " 
       << setw(fw2) << right << chunk_size << endl;
        
    os << setw(fw1) << left << "Job size" << "== " 
       << setw(fw2) << right << job_size << endl;

    os << setw(fw1) << left << "Number of W-planes" << "== " 
       << setw(fw2) << right << w_planes << endl;
  }


  void AlgorithmicParameters::print() const
  {
    print(cout);
  }


  void AlgorithmicParameters::read_parameters_from_env() 
  {
    const unsigned int DEFAULT_GRIDSIZE = 0;
    const unsigned int DEFAULT_SUBGRIDSIZE = 0;
    const unsigned int DEFAULT_CHUNKSIZE = 0;
    const unsigned int DEFAULT_JOBSIZE = 0;
    const unsigned int DEFAULT_WPLANES = 1;

    // grid_size
    char *cstr_grid_size = getenv(ENV_GRIDSIZE.c_str());
    if (cstr_grid_size != nullptr) {
      grid_size = atoi(cstr_grid_size);
    } else {
      grid_size = DEFAULT_GRIDSIZE;
    }

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
    } else {
      chunk_size = DEFAULT_CHUNKSIZE;
    }

    // job_size
    char *cstr_job_size = getenv(ENV_JOBSIZE.c_str());
    if (cstr_job_size != nullptr) {
      job_size = atoi(cstr_job_size);
    } else {
      job_size = DEFAULT_JOBSIZE;
    }

    // w_planes
    w_planes = DEFAULT_WPLANES;

  } // read_parameters_from_env()


  // helper functions
  ostream& operator<<(ostream& os, const AlgorithmicParameters& ap) 
  {
    ap.print(os);
    return os;
  }


} // namespace idg
