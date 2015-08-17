// run with -DNR_STATIONS=x -DNR_CHANNELS=... or get with arguments...

#include <iostream>
#include <cstdlib> // size_t
#include <complex>
#include "Proxy.h"
#include "Init.h"  // Data init routines
#include "Arguments.h"  // Parse command line arguments

using namespace std;

int main(int argc, char *argv[])
{
  // Set constants explicitly in the parameters parameter
  clog << ">>> Configuration"  << endl;
  idg::Parameters params;
  params.set_from_env();

  // retrieve constants for memory allocation
  int nr_stations = params.get_nr_stations();
  int nr_baselines = params.get_nr_baselines();
  int nr_timesteps = params.get_nr_timesteps();
  int nr_timeslots = params.get_nr_timeslots();
  int nr_channels = params.get_nr_channels();
  int gridsize = params.get_grid_size();
  int subgridsize = params.get_subgrid_size();
  float imagesize = params.get_imagesize();
  int nr_polarizations = 4;
  int nr_subgrids = nr_baselines * nr_timeslots;

  // Print configuration
  clog << params;
  clog << endl;

  // Allocate and initialize data structures
  clog << ">>> Initialize data structures" << endl;

  size_t size_visibilities = (size_t) nr_baselines*nr_timesteps*nr_timeslots*nr_channels*nr_polarizations;
  size_t size_uvw = (size_t) nr_baselines*nr_timesteps*nr_timeslots*3; 
  size_t size_wavenumbers = (size_t) nr_channels;
  size_t size_aterm = (size_t) nr_stations*nr_timesteps*nr_polarizations*subgridsize*subgridsize;
  size_t size_spheroidal = (size_t) subgridsize*subgridsize;
  size_t size_baselines = (size_t) nr_baselines*2;
  size_t size_grid = (size_t) nr_polarizations*gridsize*gridsize; 
  size_t size_metadata = (size_t) nr_subgrids*5;

  auto visibilities = new complex<float>[size_visibilities];
  auto uvw = new float[size_uvw];
  auto wavenumbers = new float[size_wavenumbers];
  auto aterm = new complex<float>[size_aterm];
  auto spheroidal = new float[size_spheroidal];
  auto baselines = new int[size_baselines];
  auto grid = new complex<float>[size_grid];  
  auto metadata = new int[size_metadata];

  idg::init_visibilities(visibilities, nr_baselines, nr_timesteps*nr_timeslots, nr_channels, nr_polarizations);
  idg::init_uvw(uvw, nr_stations, nr_baselines, nr_timesteps*nr_timeslots, gridsize, subgridsize);
  idg::init_wavenumbers(wavenumbers, nr_channels);
  idg::init_aterm(aterm, nr_stations, nr_timeslots, nr_polarizations, subgridsize);
  idg::init_spheroidal(spheroidal, subgridsize);
  idg::init_grid(grid, gridsize, nr_polarizations);
  idg::init_metadata(metadata, uvw, wavenumbers, nr_stations, nr_baselines, nr_timesteps, nr_timeslots, nr_channels, gridsize, subgridsize, imagesize);

  clog << endl;


  // Initialize interface to kernels
  clog << ">>> Initialize proxy" << endl;
  
  // basic gcc settings
  idg::Compiler compiler = "g++";
  idg::Compilerflags compilerflags = "-Wall -O3 -g -DDEBUG -fopenmp -lfftw3 -lfftw3f -lfftw3f_omp";

  idg::proxy::CPU xeon(compiler, compilerflags, params);
  clog << endl;

#if 0
  // Run gridding
  clog << ">>> Run gridder" << endl;
  xeon.grid_visibilities(visibilities, uvw, wavenumbers, aterm, 
  			 spheroidal, baselines, grid);
  clog << endl;

  // Run transform: Fourier->Image
  clog << ">>> Run transform" << endl;
  xeon.transform(idg::FourierDomainToImageDomain, grid);
  clog << endl;

  // do something here

  // Run transform: Image->Fourier
  clog << ">>> Run transform" << endl;
  xeon.transform(idg::ImageDomainToFourierDomain, grid);
  clog << endl;

  // Run degridding
  clog << ">>> Run degridder" << endl;
  xeon.degrid_visibilities(grid, uvw, wavenumbers, aterm, 
			   spheroidal, baselines, visibilities);
  clog << endl;
#endif


  // free memory for data structures
  delete[] visibilities;
  delete[] uvw;
  delete[] wavenumbers;
  delete[] aterm;
  delete[] spheroidal;
  delete[] baselines;
  delete[] grid;

    
  return 0;
}
