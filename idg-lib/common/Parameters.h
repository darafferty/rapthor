/** 
 *  \class ObservationParameters
 *
 *  \brief Collection of constants for a specific observation
 *
 *  Have a more detailed description here
 */

#ifndef IDG_PARAMETERS_H_
#define IDG_PARAMETERS_H_

#include <iostream>

namespace idg {

  /// Define the environment names searched for
  static const std::string ENV_NR_STATIONS = "NR_STATIONS";
  static const std::string ENV_NR_TIMESTEPS = "NR_TIME"; // for compatibility
  static const std::string ENV_NR_CHANNELS = "NR_CHANNELS";
  static const std::string ENV_NR_POLARIZATIONS = "NR_POLARIZATIONS"; // for future use
  static const std::string ENV_FIELD_OF_VIEW = "FIELD_OF_VIEW";  
  static const std::string ENV_GRIDSIZE = "GRIDSIZE";
  static const std::string ENV_WPLANES = "WPLANES"; // for future use

  /// set MIN/MAX values 
  static const float MIN_FOV = 0.0;
  static const float MAX_FOV = 1.0; // max. field of view; Q: what is the correct value here?  
  

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
    unsigned int get_nr_timesteps() const { return nr_timesteps; }
    unsigned int get_nr_channels() const { return nr_channels; }
    unsigned int get_nr_polarizations() const { return nr_polarizations; } 
    float get_field_of_view() const { return field_of_view; } 
    unsigned int get_grid_size() const { return grid_size; }
    unsigned int get_w_planes() const { return w_planes; }

    // set methods
    void set_nr_stations(unsigned int ns);
    void set_nr_timesteps(unsigned int nt);
    void set_nr_channels(unsigned int nc);
    void set_nr_polarizations(unsigned int np); // for future use
    void set_field_of_view(float fov);
    void set_grid_size(unsigned int gs);
    void set_w_planes(unsigned int wp); // for future use
 
    // auxiliary functions
    void print() const;
    void print(std::ostream& os) const;
    void read_parameters_from_env();

    static std::string 
      definitions(unsigned int nr_stations, 
		  unsigned int nr_baselines, 
		  unsigned int nr_timesteps,
		  unsigned int nr_channels, 
		  unsigned int nr_polarizations,
		  float field_of_view,
		  unsigned int grid_size,
		  unsigned int w_planes);

  private:
    unsigned int nr_stations;      
    unsigned int nr_baselines;     // nr_stations*(nr_stations-1)/2
    unsigned int nr_timesteps;     
    unsigned int nr_channels;      
    unsigned int nr_polarizations; // currently fixed to 4 
    float        field_of_view;    // unit?    
    unsigned int grid_size;
    unsigned int w_planes;         // currently fixed to 1
  };
  
  // helper functions
  std::ostream& operator<<(std::ostream& os, const Parameters& c);
  
} // namespace idg

#endif
