#include <cstdlib> // getenv, atoi
#include <iostream> // ostream
#include <iomanip> // setw
#include <sstream>
#include <cmath> // fabs

#include "idg-config.h"
#include "Parameters.h"

using namespace std;

namespace idg {

    const string Parameters::ENV_NR_STATIONS  = "NR_STATIONS";
    const string Parameters::ENV_NR_CHANNELS  = "NR_CHANNELS";
    const string Parameters::ENV_NR_TIME      = "NR_TIME";
    const string Parameters::ENV_NR_TIMESLOTS = "NR_TIMESLOTS";
    const string Parameters::ENV_IMAGESIZE    = "IMAGESIZE";
    const string Parameters::ENV_GRIDSIZE     = "GRIDSIZE";
    const string Parameters::ENV_SUBGRIDSIZE  = "SUBGRIDSIZE";
    const string Parameters::ENV_JOBSIZE      = "JOBSIZE"; // to all routines
    const string Parameters::ENV_JOBSIZE_GRIDDING   = "JOBSIZE_GRIDDING";
    const string Parameters::ENV_JOBSIZE_DEGRIDDING = "JOBSIZE_DEGRIDDING";
    const string Parameters::ENV_JOBSIZE_GRIDDER    = "JOBSIZE_GRIDDER";
    const string Parameters::ENV_JOBSIZE_ADDER      = "JOBSIZE_ADDER";
    const string Parameters::ENV_JOBSIZE_SPLITTER   = "JOBSIZE_SPLITTER";
    const string Parameters::ENV_JOBSIZE_DEGRIDDER  = "JOBSIZE_DEGRIDDER";


  // set methods
  void Parameters::set_nr_stations(unsigned int ns)
  {
    nr_stations = ns > 1 ? ns : 1;
    nr_baselines = (nr_stations * (nr_stations-1)) / 2;
  }

  void Parameters::set_nr_channels(unsigned int nc)
  {
    nr_channels = nc > 0 ? nc : 1;
  }

  void Parameters::set_nr_time(unsigned int nt)
  {
    nr_time = nt > 0 ? nt : 1;
  }

  void Parameters::set_nr_timeslots(unsigned int nt)
  {
    nr_timeslots = nt > 0 ? nt : 1;
  }

  void Parameters::set_imagesize(float is)
  {
    imagesize = is < 0 ? 1 : is;
  }

  void Parameters::set_grid_size(unsigned int gs)
  {
    grid_size = gs;
  }

  void Parameters::set_subgrid_size(unsigned int sgs)
  {
    subgrid_size = sgs;
  }

  void Parameters::set_job_size(unsigned int js)
  {
    if (js == 0) return; // avoid job_size==0
    job_size = js;
    job_size_gridding = js;
    job_size_degridding = js;
    job_size_gridder = js;
    job_size_adder = js;
    job_size_splitter = js;
    job_size_degridder = js;
  }

  void Parameters::set_job_size_gridding(unsigned int js)
  {
    if (js == 0) return; // avoid job_size==0
    job_size_gridding = js;
    job_size_gridder = js;
    job_size_adder = js;
  }

  void Parameters::set_job_size_degridding(unsigned int js)
  {
    if (js == 0) return; // avoid job_size==0
    job_size_degridding = js;
    job_size_splitter = js;
    job_size_degridder = js;
  }

  void Parameters::set_job_size_gridder(unsigned int js)
  {
    if(js > 0) job_size_gridder = js;
  }

  void Parameters::set_job_size_adder(unsigned int js)
  {
    if(js > 0) job_size_adder = js;
  }

  void Parameters::set_job_size_splitter(unsigned int js)
  {
    if(js > 0) job_size_splitter = js;
  }

  void Parameters::set_job_size_degridder(unsigned int js)
  {
    if(js > 0) job_size_degridder = js;
  }

  // auxiliary functions
  void Parameters::print(ostream& os) const
  {
    const int fw1 = 30;
    const int fw2 = 10;

    os << "-----------" << endl;
    os << "PARAMETERS:" << endl;

    os << setw(fw1) << left << "Number of stations" << "== "
       << setw(fw2) << right << nr_stations << endl;

    os << setw(fw1) << left << "Number of baselines" << "== "
       << setw(fw2) << right << nr_baselines << endl;

    os << setw(fw1) << left << "Number of channels" << "== "
       << setw(fw2) << right << nr_channels << endl;

    os << setw(fw1) << left << "Number of time" << "== "
       << setw(fw2) << right << nr_time << endl;

    os << setw(fw1) << left << "Number of timeslots" << "== "
       << setw(fw2) << right << nr_timeslots << endl;

    os << setw(fw1) << left << "Imagesize" << "== "
       << setw(fw2) << right << imagesize  << endl;

    os << setw(fw1) << left << "Grid size" << "== "
       << setw(fw2) << right << grid_size << endl;

    os << setw(fw1) << left << "Subgrid size" << "== "
       << setw(fw2) << right << subgrid_size << endl;

    os << setw(fw1) << left << "Job size" << "== "
       << setw(fw2) << right << job_size << endl;

    os << setw(fw1) << left << "Job size (gridding)" << "== "
       << setw(fw2) << right << job_size_gridding << endl;

    os << setw(fw1) << left << "Job size (degridding)" << "== "
       << setw(fw2) << right << job_size_degridding << endl;

    os << setw(fw1) << left << "Job size (gridder)" << "== "
       << setw(fw2) << right << job_size_gridder << endl;

    os << setw(fw1) << left << "Job size (adder)" << "== "
       << setw(fw2) << right << job_size_adder << endl;

    os << setw(fw1) << left << "Job size (splitter)" << "== "
       << setw(fw2) << right << job_size_splitter << endl;

    os << setw(fw1) << left << "Job size (degridder)" << "== "
       << setw(fw2) << right << job_size_degridder << endl;

    os << "-----------" << endl;
  }


  void Parameters::print() const
  {
    print(cout);
  }


  void Parameters::set_from_env()
  {
    const unsigned int DEFAULT_NR_STATIONS = 44;
    const unsigned int DEFAULT_NR_CHANNELS = 8;
    const unsigned int DEFAULT_NR_TIME = 2048;
    const unsigned int DEFAULT_NR_TIMESLOTS = 128;
    const unsigned int DEFAULT_NR_POLARIZATIONS = 4;
    const float DEFAULT_IMAGESIZE = 0.1f;
    const unsigned int DEFAULT_GRIDSIZE = 4096;
    const unsigned int DEFAULT_SUBGRIDSIZE = 32;
    const unsigned int DEFAULT_JOBSIZE = 256;

    // nr_stations
    char *cstr_nr_stations = getenv(ENV_NR_STATIONS.c_str());
    nr_stations = cstr_nr_stations ? atoi(cstr_nr_stations): DEFAULT_NR_STATIONS;

    // nr_baselines
    nr_baselines = (nr_stations * (nr_stations-1)) / 2;

    // nr_channels
    char *cstr_nr_channels = getenv(ENV_NR_CHANNELS.c_str());
    nr_channels = cstr_nr_channels ? atoi(cstr_nr_channels) : DEFAULT_NR_CHANNELS;

    // nr_time
    char *cstr_nr_time = getenv(ENV_NR_TIME.c_str());
    nr_time = cstr_nr_time ? atoi(cstr_nr_time) : DEFAULT_NR_TIME;

    // nr_timeslots
    char *cstr_nr_timeslots = getenv(ENV_NR_TIMESLOTS.c_str());
    nr_timeslots = cstr_nr_timeslots ? atoi(cstr_nr_timeslots) : DEFAULT_NR_TIMESLOTS;

    // imagesize
    char *cstr_imagesize = getenv(ENV_IMAGESIZE.c_str());
    imagesize = cstr_imagesize ? atof(cstr_imagesize) : DEFAULT_IMAGESIZE;

    // grid_size
    char *cstr_grid_size = getenv(ENV_GRIDSIZE.c_str());
    grid_size = cstr_grid_size ? atoi(cstr_grid_size) : DEFAULT_GRIDSIZE;

    // subgrid_size
    char *cstr_subgrid_size = getenv(ENV_SUBGRIDSIZE.c_str());
    subgrid_size = cstr_subgrid_size ? atoi(cstr_subgrid_size) : DEFAULT_SUBGRIDSIZE;

    // job_size
    char *cstr_job_size = getenv(ENV_JOBSIZE.c_str());
    job_size = cstr_job_size ? atoi(cstr_job_size) : DEFAULT_JOBSIZE;

    // job_size_*
    char *cstr_job_size_gridding = getenv(ENV_JOBSIZE_GRIDDING.c_str());
    job_size_gridding = cstr_job_size_gridding ? atoi(cstr_job_size_gridding) : job_size;

    char *cstr_job_size_degridding = getenv(ENV_JOBSIZE_DEGRIDDING.c_str());
    job_size_degridding = cstr_job_size_degridding ? atoi(cstr_job_size_degridding) : job_size;

    char *cstr_job_size_gridder = getenv(ENV_JOBSIZE_GRIDDER.c_str());
    job_size_gridder = cstr_job_size_gridder ? atoi(cstr_job_size_gridder) : job_size;

    char *cstr_job_size_adder = getenv(ENV_JOBSIZE_ADDER.c_str());
    job_size_adder = cstr_job_size_adder ? atoi(cstr_job_size_adder) : job_size;

    char *cstr_job_size_splitter = getenv(ENV_JOBSIZE_SPLITTER.c_str());
    job_size_splitter = cstr_job_size_splitter ? atoi(cstr_job_size_splitter) : job_size;

    char *cstr_job_size_degridder = getenv(ENV_JOBSIZE_DEGRIDDER.c_str());
    job_size_degridder = cstr_job_size_degridder ? atoi(cstr_job_size_degridder) : job_size;
  } // read_parameters_from_env()


  string Parameters::definitions(
            unsigned int nr_stations,
            unsigned int nr_baselines,
            unsigned int nr_channels,
            unsigned int nr_timeslots,
            float imagesize,
            unsigned int nr_polarizations,
            unsigned int grid_size,
            unsigned int subgrid_size) {
    stringstream parameters;
    parameters << " -DNR_STATIONS=" << nr_stations;
    parameters << " -DNR_BASELINES=" << nr_baselines;
    parameters << " -DNR_CHANNELS=" << nr_channels;
    parameters << " -DNR_TIMESLOTS=" << nr_timeslots;
    parameters << " -DIMAGESIZE=" << imagesize;
    parameters << " -DNR_POLARIZATIONS=" << nr_polarizations;
    parameters << " -DGRIDSIZE=" << grid_size;
    parameters << " -DSUBGRIDSIZE=" << subgrid_size;
    return parameters.str();
  }

  // helper functions
  ostream& operator<<(ostream& os, const Parameters& c)
  {
    c.print(os);
    return os;
  }

} // namespace idg
