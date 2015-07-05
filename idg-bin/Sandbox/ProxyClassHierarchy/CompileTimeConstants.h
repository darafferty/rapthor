
#ifndef IDG_COMPILETIMECONSTANTS_H_
#define IDG_COMPILETIMECONSTANTS_H_

#include <iostream>
#include "ObservationParameters.h"
#include "AlgorithmicParameters.h"

namespace idg {

  class CompileTimeConstants 
  {
  public:
    /// Constructor: default reads values from ENV or sets default 
    CompileTimeConstants() = default;
    CompileTimeConstants(ObservationParameters& op, 
    			 AlgorithmicParameters& ap) 
      : observation_parameters(op),
      algorithmic_parameters(ap) {};
  
    // default copy constructor/assignment okay
    
    // default destructur
    ~CompileTimeConstants() = default;
    
    /// Set the number of stations in [1,UINT_MAX]
    /** A more detailed description could be here */
    void set_nr_stations(unsigned int ns) { 
      observation_parameters.set_nr_stations(ns); 
    } 
    
    /// Set the number of timesteps in [1,UINT_MAX]
    /** A more detailed description could be here */
    void set_nr_timesteps(unsigned int nt) { 
      observation_parameters.set_nr_timesteps(nt); 
    }
    
    /// Set the number of channels in [1,UINT_MAX]
    /** A more detailed description could be here */
    void set_nr_channels(unsigned int nc) { 
      observation_parameters.set_nr_channels(nc); 
    }
    
    /// Set the number of polarizations in [1,2,4], only 4 supported so far
    /** A more detailed description could be here */
    void set_nr_polarizations(unsigned int np) { 
      observation_parameters.set_nr_polarizations(np); 
    }
    
    /// Set the field of view in (unit) [0,MAX_FOV]
    /** A more detailed description could be here */
    void set_field_of_view(float fov) { 
      observation_parameters.set_field_of_view(fov); 
    }
    
    /// Set the grid size N: constructed image is N-by-N 
    /** A more detailed description could be here */
    void set_grid_size(unsigned int gs) { 
      algorithmic_parameters.set_grid_size(gs); 
    }

    /// Set the subgrid used in image domain gridding
    /** A more detailed description should be there */
    void set_subgrid_size(unsigned int sgs) { 
      algorithmic_parameters.set_subgrid_size(sgs); 
    }
    
    /// Set the chunk size used in image domain gridding
    /** A more detailed description should be there */
    void set_chunk_size(unsigned int cs) { 
      algorithmic_parameters.set_chunk_size(cs); 
    }
    
    /// Set the job size used in image domain gridding
    /** A more detailed description should be there */
    void set_job_size(unsigned int js) {
      algorithmic_parameters.set_job_size(js); 
    } 

    /// Set the number of W-planes used in image domain gridding
    /** A more detailed description should be there */
    void set_w_planes(unsigned int wp) {
      algorithmic_parameters.set_w_planes(wp);
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

    unsigned int get_w_planes() const { 
      return algorithmic_parameters.get_w_planes(); 
    }
    
    
    // display all parameters    
    void print(std::ostream& os) const { 
      os << "-----------------------" << std::endl;
      os << "COMPILE TIME CONSTANTS:" << std::endl;
      os << std::endl;
      observation_parameters.print(os);
      os << std::endl;
      algorithmic_parameters.print(os); 
      os << "-----------------------" << std::endl;
    }

    void print() const { 
      print(std::cout);
    }    

  private:
    ObservationParameters observation_parameters; 
    AlgorithmicParameters algorithmic_parameters;
  };
  
  // helper functions
  std::ostream& operator<<(std::ostream& os, const CompileTimeConstants& ctc);
  
} // namespace idg

#endif
