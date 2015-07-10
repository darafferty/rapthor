#include <cstdlib> // getenv, atoi
#include <iostream> // ostream
#include <iomanip> // setw
#include <sstream>
#include <cmath> // fabs
#include "Parameters.h"

using namespace std;

namespace idg {

  // set methods
  void Parameters::set_nr_stations(unsigned int ns) 
  {
    if (ns > 1) {
      nr_stations = ns;
    } else {
      nr_stations = 1;
    }
    nr_baselines = (nr_stations * (nr_stations-1)) / 2;
  }
  
  void Parameters::set_nr_timesteps(unsigned int nt) 
  {
    if (nt > 0) {
      nr_timesteps = nt;
    } else {
      nr_timesteps = 1;
    }
  }

  void Parameters::set_nr_channels(unsigned int nc) 
  {
    if (nc > 0) {
      nr_channels = nc;
    } else {
      nr_channels = 1;
    }
  }

  void Parameters::set_nr_polarizations(unsigned int np) 
  {
    if (np != 4) 
      cerr << "WARNING: Setting the number of polarizations is currently not supported." << endl;    
  }

  void Parameters::set_field_of_view(float fov) 
  {
    if (fov < MIN_FOV) {
      field_of_view = MIN_FOV;
    } else if (fov > MAX_FOV) {
      field_of_view = MAX_FOV;
    } else {
      field_of_view = fov;
    }
  }

  void Parameters::set_grid_size(unsigned int gs)
  {
    grid_size = gs;
  }

  void Parameters::set_w_planes(unsigned int wp) 
  { 
    if (wp != 1)
      cerr << "WARNING: Setting the number of W-planes is currently not supported." << endl;    
  }


  // auxiliary functions
  void Parameters::print(ostream& os) const
  {
    const int fw1 = 30;
    const int fw2 = 10;
    const int fw3 = 10;

    os << "OBSERVATION PARAMETERS:" << endl;
    
    os << setw(fw1) << left << "Number of stations" << "== " 
       << setw(fw2) << right << nr_stations << endl;
    
    os << setw(fw1) << left << "Number of baselines" << "== " 
       << setw(fw2) << right << nr_baselines << endl;
    
    os << setw(fw1) << left << "Number of timesteps" << "== " 
       << setw(fw2) << right << nr_timesteps << endl;
    
    os << setw(fw1) << left << "Number of channels" << "== " 
       << setw(fw2) << right << nr_channels << endl;
    
    os << setw(fw1) << left << "Number of polarizations" << "== " 
       << setw(fw2) << right << nr_polarizations << endl;
    
    os << setw(fw1) << left << "Field of view" << "== " 
       << setw(fw2) << right <<  field_of_view 
       << setw(fw3) << right << "(unit)" << endl;

    os << setw(fw1) << left << "Grid size" << "== " 
       << setw(fw2) << right << grid_size << endl;
    
    os << setw(fw1) << left << "Number of W-planes" << "== " 
       << setw(fw2) << right << w_planes << endl;
  }


  void Parameters::print() const
  {
    print(cout);
  }
 

  void Parameters::read_parameters_from_env() 
  {
    const unsigned int DEFAULT_NR_STATIONS = 0;
    const unsigned int DEFAULT_NR_TIMESTEPS = 0;
    const unsigned int DEFAULT_NR_CHANNELS = 0;
    const unsigned int DEFAULT_NR_POLARIZATIONS = 4;
    const float DEFAULT_FIELD_OF_VIEW = 0.0f;
    const unsigned int DEFAULT_GRIDSIZE = 0;
    const unsigned int DEFAULT_WPLANES = 1;

    // nr_stations
    char *cstr_nr_stations = getenv(ENV_NR_STATIONS.c_str());
    if (cstr_nr_stations != nullptr) {
      nr_stations = atoi(cstr_nr_stations);
    } else {
      nr_stations = DEFAULT_NR_STATIONS;
    }

    // nr_baselines 
    nr_baselines = (nr_stations * (nr_stations-1)) / 2;

    // nr_timesteps
    char *cstr_nr_timesteps = getenv(ENV_NR_TIMESTEPS.c_str());
    if (cstr_nr_timesteps != nullptr) {
      nr_timesteps = atoi(cstr_nr_timesteps);
    } else {
      nr_timesteps = DEFAULT_NR_TIMESTEPS;
    }

    // nr_channels
    char *cstr_nr_channels = getenv(ENV_NR_CHANNELS.c_str());
    if (cstr_nr_channels != nullptr) {
      nr_channels = atoi(cstr_nr_channels);
    } else {
      nr_channels = DEFAULT_NR_CHANNELS;
    }

    // nr_polarizations
    nr_polarizations = DEFAULT_NR_POLARIZATIONS;

    // field_of_view
    char *cstr_fov = getenv(ENV_FIELD_OF_VIEW.c_str());
    if (cstr_fov != nullptr) {
      field_of_view = atof(cstr_fov);
    } else {
      field_of_view = DEFAULT_FIELD_OF_VIEW;
    }

    // grid_size
    char *cstr_grid_size = getenv(ENV_GRIDSIZE.c_str());
    if (cstr_grid_size != nullptr) {
      grid_size = atoi(cstr_grid_size);
    } else {
      grid_size = DEFAULT_GRIDSIZE;
    }

    // w_planes
    w_planes = DEFAULT_WPLANES;

  } // read_parameters_from_env()


  string Parameters::definitions(unsigned int nr_stations, 
				 unsigned int nr_baselines, 
				 unsigned int nr_timesteps,
				 unsigned int nr_channels, 
				 unsigned int nr_polarizations,
				 float field_of_view,
				 unsigned int grid_size,
				 unsigned int w_planes) {
    stringstream parameters;
    parameters << " -DNR_STATIONS=" << nr_stations;
    parameters << " -DNR_BASELINES=" << nr_baselines;
    parameters << " -DNR_TIME=" << nr_timesteps;
    parameters << " -DNR_CHANNELS=" << nr_channels;
    parameters << " -DNR_POLARIZATIONS=" << nr_polarizations;
    parameters << " -DIMAGESIZE=" << field_of_view;
    parameters << " -DGRIDSIZE=" << grid_size;
    parameters << " -DWPLANES=" << w_planes;
    return parameters.str();
  }


  // helper functions
  ostream& operator<<(ostream& os, const Parameters& c) 
  {
    c.print(os);
    return os;
  }

} // namespace idg
