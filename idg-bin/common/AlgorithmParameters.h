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
  static const std::string ENV_SUBGRIDSIZE = "SUBGRIDSIZE"; 
  static const std::string ENV_CHUNKSIZE = "CHUNKSIZE";
  static const std::string ENV_JOBSIZE = "JOBSIZE"; // to all routines

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

    // get methods
    unsigned int get_subgrid_size() const { return subgrid_size; }
    unsigned int get_chunk_size() const { return chunk_size; }
    unsigned int get_job_size() const { return job_size; }

    unsigned int get_job_size_gridding() const { return job_size_gridding; }
    unsigned int get_job_size_degridding() const { return job_size_degridding; }

    unsigned int get_job_size_gridder() const { return job_size_gridder; } 
    unsigned int get_job_size_adder() const { return job_size_adder; } 

    unsigned int get_job_size_splitter() const { return job_size_splitter; }
    unsigned int get_job_size_degridder() const { return job_size_degridder; }

    // set methods    
    void set_subgrid_size(unsigned int sgs);
    void set_chunk_size(unsigned int cs);
    void set_job_size(unsigned int js); 

    void set_job_size_gridding(unsigned int js); 
    void set_job_size_degridding(unsigned int js); 

    void set_job_size_gridder(unsigned int js); 
    void set_job_size_adder(unsigned int js); 

    void set_job_size_splitter(unsigned int js); 
    void set_job_size_degridder(unsigned int js); 
    
    
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
    unsigned int job_size; // rename? 
    unsigned int job_size_gridding;
    unsigned int job_size_degridding;
    unsigned int job_size_gridder; 
    unsigned int job_size_adder; 
    unsigned int job_size_splitter; 
    unsigned int job_size_degridder; 
  };

  // helper functions
  std::ostream& operator<<(std::ostream& os, const AlgorithmParameters& ap);
  
} // namespace idg

#endif
