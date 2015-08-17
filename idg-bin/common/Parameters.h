/** 
 *  \class Parameters
 *
 *  \brief Collection of constants for a specific invocation of IDG
 *
 *  Have a more detailed description here
 */

#ifndef IDG_PARAMETERS_H_
#define IDG_PARAMETERS_H_

#include <iostream>

namespace idg {

  /// Define the environment names searched for
  static const std::string ENV_NR_STATIONS  = "NR_STATIONS";
  static const std::string ENV_NR_CHANNELS  = "NR_CHANNELS";
  static const std::string ENV_NR_TIMESTEPS = "NR_TIMESLOTS";
  static const std::string ENV_NR_TIMESLOTS = "NR_TIMESTEPS";
  static const std::string ENV_IMAGESIZE    = "IMAGESIZE";  
  static const std::string ENV_GRIDSIZE     = "GRIDSIZE";
  static const std::string ENV_SUBGRIDSIZE  = "SUBGRIDSIZE";
  static const std::string ENV_JOBSIZE      = "JOBSIZE"; // to all routines

  class Parameters 
  {
  public:
    /// Constructor: default reads values from ENV or sets default 
    Parameters() {
      read_parameters_from_env();
    }
  
    // default copy constructor/assignment okay
    
    // default destructur
    ~Parameters() = default;

    // get methods
    unsigned int get_nr_stations() const { return nr_stations; }
    unsigned int get_nr_baselines() const { return nr_baselines; }
    unsigned int get_nr_channels() const { return nr_channels; }
    unsigned int get_nr_timesteps() const { return nr_timesteps; }
    unsigned int get_nr_timeslots() const { return nr_timeslots; }
    float get_imagesize() const { return imagesize; } 
    unsigned int get_grid_size() const { return grid_size; }
    unsigned int get_subgrid_size() const { return subgrid_size; }
    unsigned int get_job_size() const { return job_size; }
    unsigned int get_job_size_gridding() const { return job_size_gridding; }
    unsigned int get_job_size_degridding() const { return job_size_degridding; }
    unsigned int get_job_size_gridder() const { return job_size_gridder; }
    unsigned int get_job_size_adder() const { return job_size_adder; }
    unsigned int get_job_size_splitter() const { return job_size_splitter; }
    unsigned int get_job_size_degridder() const { return job_size_degridder; }
    unsigned int get_nr_polarizations() const { return nr_polarizations; }

    // set methods
    void set_nr_stations(unsigned int ns);
    void set_nr_channels(unsigned int nc);
    void set_nr_timesteps(unsigned int nt);
    void set_nr_timeslots(unsigned int nt);
    void set_imagesize(float imagesize);
    void set_subgrid_size(unsigned int sgs);
    void set_grid_size(unsigned int gs);
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

    static std::string 
      definitions(
          unsigned int nr_stations, 
          unsigned int nr_baselines,
		  unsigned int nr_channels, 
		  unsigned int nr_timesteps,
		  unsigned int nr_timeslots,
		  float imagesize, 
		  unsigned int grid_size,
          unsigned int subgrid_size);

  private:
    unsigned int nr_stations;      
    unsigned int nr_baselines;     // nr_stations*(nr_stations-1)/2
    unsigned int nr_channels;      
    unsigned int nr_timesteps;     
    unsigned int nr_timeslots;     
    unsigned int nr_polarizations = 4;
    float        imagesize;        // angular resolution in radians
    unsigned int grid_size;
    unsigned int subgrid_size;
    unsigned int job_size;
    unsigned int job_size_gridding;
    unsigned int job_size_degridding;
    unsigned int job_size_gridder; 
    unsigned int job_size_adder; 
    unsigned int job_size_splitter; 
    unsigned int job_size_degridder; 
  };
  
  // helper functions
  std::ostream& operator<<(std::ostream& os, const Parameters& c);
  
} // namespace idg

#endif
