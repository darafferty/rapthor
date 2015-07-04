
#ifndef IDG_COMPILETIMECONSTANTS_H_
#define IDG_COMPILETIMECONSTANTS_H_

#include <iostream>
#include "ObservationParameters.h"
#include "AlgorithmicParameters.h"

namespace idg {

  class CompileTimeConstants 
  {
  public:
    // default constructor reads from parameters from ENV
    //  CompileTimeConstants() {};
    // CompileTimeConstants(ObservationParameters& op, 
    //			AlgorithmicParameters& ap);
  
    // default copy constructor/assignment okay
    
    // default destructur
    ~CompileTimeConstants() {};
    
    // set and get methods
    void set_nr_stations(unsigned int ns) { 
      observation_parameters.set_nr_stations(ns); 
    } 
    
    void set_nr_timesteps(unsigned int nt) { 
      observation_parameters.set_nr_timesteps(nt); 
    }
    
    void set_nr_channels(unsigned int nc) { 
      observation_parameters.set_nr_channels(nc); 
    }
    
    void set_nr_polarizations(unsigned int np) { 
      observation_parameters.set_nr_polarizations(np); 
    }
    
    void set_field_of_view(float fov) { 
      observation_parameters.set_field_of_view(fov); 
    }
    
    void set_grid_size(unsigned int gs) { 
      algorithmic_parameters.set_grid_size(gs); 
    }
    
    void set_subgrid_size(unsigned int sgs) { 
      algorithmic_parameters.set_subgrid_size(sgs); 
    }
    
    void set_chunk_size(unsigned int cs) { 
      algorithmic_parameters.set_chunk_size(cs); 
    }
    
    void set_job_size(unsigned int js) {
      algorithmic_parameters.set_job_size(js); 
    } 
    
    
    unsigned int get_nr_stations() const { 
      return observation_parameters.get_nr_stations(); 
    }
    
    unsigned int get_nr_timesteps() const { 
      return observation_parameters.get_nr_timesteps(); 
    }
    
    unsigned int get_nr_channels() const { 
      return observation_parameters.get_nr_channels(); 
    }
    
    unsigned int get_nr_polarizations() const { 
      return observation_parameters.get_nr_polarizations(); 
    }
    
    float get_field_of_view() const { 
      return observation_parameters.get_field_of_view(); 
    } 
    
    unsigned int get_grid_size() const { 
      return algorithmic_parameters.get_grid_size(); 
    }
    
    unsigned int get_subgrid_size() const { 
      return algorithmic_parameters.get_subgrid_size(); 
    }
    
    unsigned int get_chunk_size() const { 
      return algorithmic_parameters.get_chunk_size(); 
    }
    
    unsigned int get_job_size() const { 
      return algorithmic_parameters.get_job_size(); 
    }
    
    
    // display all parameters
    void print() const { 
      observation_parameters.print(); 
      std::cout << std::endl;
      algorithmic_parameters.print(); 
    }
    
    void print(std::ostream& os) const { 
      observation_parameters.print(os);
      os << std::endl;
      algorithmic_parameters.print(os); 
    }
    
  private:
    ObservationParameters observation_parameters;
    AlgorithmicParameters algorithmic_parameters;
  };
  
  // helper functions
  std::ostream& operator<<(std::ostream& os, const CompileTimeConstants& ctc);
  
} // namespace idg

#endif
