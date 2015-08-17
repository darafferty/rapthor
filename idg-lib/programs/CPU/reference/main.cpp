// This file mimics IDG-WRAPPER
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
  // Get parameters passed as arguments
  int nr_stations = 0;
  int nr_time = 0;
  int nr_channels = 0;
  int w_planes = 0;
  int gridsize = 0;
  int subgridsize = 0;
  int chunksize = 0;
  int jobsize = 0;
  idg::get_parameters(argc, argv, &nr_stations, &nr_time, &nr_channels, 
                      &w_planes, &gridsize, &subgridsize, &chunksize, &jobsize);

  // Set other parameters
  int nr_polarizations = 4;

  // Set constants explicitly in the parameters parameter
  clog << ">>> Configuration"  << endl;
  idg::Parameters params;    
  params.set_nr_stations(nr_stations);
  params.set_nr_timesteps(nr_time);
  params.set_nr_channels(nr_channels);
  params.set_grid_size(gridsize);

  // retrieve constants for memory allocation
  nr_stations = params.get_nr_stations();
  auto nr_baselines = params.get_nr_baselines();
  nr_time = params.get_nr_timesteps();
  nr_channels = params.get_nr_channels();
  gridsize = params.get_grid_size();

  //  int jobsize = algparams.get_job_size();
  subgridsize = params.get_subgrid_size();

  // Print configuration
  clog << params;
  clog << endl;

  // Allocate and initialize data structures
  clog << ">>> Initialize data structures" << endl;

  size_t size_visibilities = (size_t) nr_baselines*nr_time*nr_channels*nr_polarizations;
  size_t size_uvw = (size_t) nr_baselines*nr_time*3; 
  size_t size_wavenumbers = (size_t) nr_channels;
  size_t size_aterm = (size_t) nr_stations*nr_polarizations*subgridsize*subgridsize;
  size_t size_spheroidal = (size_t) subgridsize*subgridsize;
  size_t size_baselines = (size_t) nr_baselines*2;
  size_t size_grid = (size_t) nr_polarizations*gridsize*gridsize; 

  auto visibilities = new complex<float>[size_visibilities];
  auto uvw = new float[size_uvw];
  auto wavenumbers = new float[size_wavenumbers];
  auto aterm = new complex<float>[size_aterm];
  auto spheroidal = new float[size_spheroidal];
  auto baselines = new int[size_baselines];
  auto grid = new complex<float>[size_grid];  

  idg::init_visibilities(visibilities, nr_baselines, nr_time, nr_channels, 
                         nr_polarizations);
  idg::init_uvw(uvw, nr_stations, nr_baselines, nr_time, gridsize, 
                subgridsize, w_planes);
  idg::init_wavenumbers(wavenumbers, nr_channels);
  idg::init_aterm(aterm, nr_stations, nr_polarizations, subgridsize);
  idg::init_spheroidal(spheroidal, subgridsize);
  idg::init_baselines(baselines, nr_stations, nr_baselines);
  idg::init_grid(grid, gridsize, nr_polarizations);

  clog << endl;


  // Initialize interface to kernels
  clog << ">>> Initialize proxy" << endl;
  
  // basic gcc settings
  idg::Compiler compiler = "g++";
  idg::Compilerflags compilerflags = "-Wall -O3 -g -DDEBUG -fopenmp -lfftw3 -lfftw3f -lfftw3f_omp";

#if 0
  idg::proxy::CPU xeon(compiler, compilerflags, params, algparams);
  clog << endl;


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
