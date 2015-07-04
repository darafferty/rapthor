
#ifndef IDG_ALGORITHMICPARAMETERS_H_
#define IDG_ALGORITHMICPARAMETERS_H_

#include <iostream>

namespace idg {

  // define the enviroment names searched for
  const std::string ENV_GRIDSIZE = "GRIDSIZE";
  const std::string ENV_SUBGRIDSIZE = "SUBGRIDSIZE"; 
  const std::string ENV_CHUNKSIZE = "CHUNKSIZE";
  const std::string ENV_JOBSIZE = "JOBSIZE";
  const std::string ENV_WPLANES = "WPLANES"; // for future use

  class AlgorithmicParameters 
  {
  public:
    // default constructor reads from parameters from ENV
    AlgorithmicParameters() {
      read_parameters_from_env();
    }
    
    // default copy constructor/assignment okay
    
    // default destructur
    ~AlgorithmicParameters() = default;
    
    // set and get methods
    void set_grid_size(unsigned int gs);
    void set_subgrid_size(unsigned int sgs);
    void set_chunk_size(unsigned int cs);
    void set_job_size(unsigned int js); 
    void set_w_planes(unsigned int wp); // for future use
    
    unsigned int get_grid_size() const { return grid_size; }
    unsigned int get_subgrid_size() const { return subgrid_size; }
    unsigned int get_chunk_size() const { return chunk_size; }
    unsigned int get_job_size() const { return job_size; }
    unsigned int get_w_planes() const { return w_planes; }
    
    // auxiliary functions
    void print() const;
    void print(std::ostream& os) const;
    void read_parameters_from_env();
    
  private:
    unsigned int grid_size;
    unsigned int subgrid_size;
    unsigned int chunk_size; // rename?
    unsigned int job_size; // rename?
    unsigned int w_planes;    
  };

  // helper functions
  std::ostream& operator<<(std::ostream& os, const AlgorithmicParameters& ap);
  
} // namespace idg

#endif
