
#ifndef IDG_OBSERVATIONPARAMETERS_H_
#define IDG_OBSERVATIONPARAMETERS_H_

#include <iostream>
#include <iostream>

namespace idg {

  // define the enviroment names searched for
  const std::string ENV_NR_STATIONS = "NR_STATIONS";
  const std::string ENV_NR_TIMESTEPS = "NR_TIME"; // for compatibility
  const std::string ENV_NR_CHANNELS = "NR_CHANNELS";
  const std::string ENV_NR_POLARIZATIONS = "NR_POLARIZATIONS"; // for future use
  const std::string ENV_FIELD_OF_VIEW = "FIELD_OF_VIEW";  

  class ObservationParameters 
  {
  public:
    // default constructor read from enviroment
    ObservationParameters() {
      read_parameters_from_env();
    }
    
    // default copy constructor/assignment okay
    
    // default destructur
    ~ObservationParameters() {};
    
    // set and get methods
    void set_nr_stations(unsigned int ns);
    void set_nr_timesteps(unsigned int nt);
    void set_nr_channels(unsigned int nc);
    void set_nr_polarizations(unsigned int np); // for future use
    void set_field_of_view(float fov);

    unsigned int get_nr_stations() const { return nr_stations; }
    unsigned int get_nr_timesteps() const { return nr_timesteps; }
    unsigned int get_nr_channels() const { return nr_channels; }
    unsigned int get_nr_polarizations() const { return nr_polarizations; } 
    float get_field_of_view() const { return field_of_view; } 
        
    // auxiliary functions
    void print() const;
    void print(std::ostream& os) const;
    void read_parameters_from_env();
    
  private:
    unsigned int nr_stations;
    unsigned int nr_baselines;
    unsigned int nr_timesteps;
    unsigned int nr_channels;
    unsigned int nr_polarizations; 
    float field_of_view;           // unit?    
  };

  // helper functions
  std::ostream& operator<<(std::ostream& os, const ObservationParameters& op);

} // namespace idg

#endif
