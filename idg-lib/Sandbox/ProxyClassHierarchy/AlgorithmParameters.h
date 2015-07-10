/** 
 *  \class AlgorithmParameters
 *
 *  \brief Collection of constants for a specific invocation of IDG
 *
 *  Have a more detailed description here
 */

#ifndef IDG_ALGORITHMPARAMETERS_H_
#define IDG_ALGORITHMPARAMETERS_H_

#include <iostream>

namespace idg {

  /// Define the environment names searched for
  const std::string ENV_SUBGRIDSIZE = "SUBGRIDSIZE"; 
  const std::string ENV_CHUNKSIZE = "CHUNKSIZE";
  const std::string ENV_JOBSIZE = "JOBSIZE"; // to all routines

  class AlgorithmParameters 
  {
  public:
    /// Default constructor reads from parameters from ENV
    AlgorithmParameters() {
      read_parameters_from_env();
    }
    
    // default copy constructor/assignment okay
    
    // default destructur
    ~AlgorithmParameters() = default;
    
    // set and get methods    
    void set_subgrid_size(unsigned int sgs);
    void set_chunk_size(unsigned int cs);
    void set_job_size(unsigned int js); 
    
    unsigned int get_subgrid_size() const { return subgrid_size; }
    unsigned int get_chunk_size() const { return chunk_size; }
    unsigned int get_job_size() const { return job_size; }
    
    // auxiliary functions
    void print() const;
    void print(std::ostream& os) const;
    void read_parameters_from_env();

    static std::string definitions(unsigned int subgrid_size,
				   unsigned int chunk_size,
				   unsigned int job_size);
    
  private:
    unsigned int subgrid_size;
    unsigned int chunk_size; // rename?
    unsigned int job_size; // rename? THIS SEEMS TO BE NOT A COMPILE TIME CONSTANT, BUT RUNTIME PARAMETER!
  };

  // helper functions
  std::ostream& operator<<(std::ostream& os, const AlgorithmParameters& ap);
  
} // namespace idg

#endif
