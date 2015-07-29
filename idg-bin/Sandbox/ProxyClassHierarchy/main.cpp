// This file mimics IDG-WRAPPER
// run with -DNR_STATIONS=x -DNR_CHANNELS=... or get with arguments...

#include <iostream>
#include <cstdlib> // size_t
#include <complex>
#include "SMP.h"
#include "Init.h"  // Data init routines

using namespace std;

int main(int argc, char *argv[])
{
  // Set constants explicitly
  idg::Parameters params;    
  params.set_nr_stations(24);
  params.set_nr_timesteps(256);
  params.set_nr_channels(32);
  params.set_nr_polarizations(4); 
  params.set_field_of_view(0.1);
  params.set_grid_size(128);
  params.set_w_planes(1);

  idg::AlgorithmParameters algparams;
  algparams.set_job_size(8); 
  algparams.set_subgrid_size(8); 
  algparams.set_chunk_size(8); 
  
  // retrieve constants for memory allocation
  int nr_stations = params.get_nr_stations();
  int nr_baselines = params.get_nr_baselines();
  int nr_time = params.get_nr_timesteps();
  int nr_channels = params.get_nr_channels();
  int nr_polarizations = params.get_nr_polarizations();
  int w_planes = params.get_w_planes();
  int gridsize = params.get_grid_size();

  //  int jobsize = algparams.get_job_size();
  int subgridsize = algparams.get_subgrid_size();

  // Print configuration
  clog << ">>> Configuration"  << std::endl;
  clog << params;
  clog << endl;

  // Allocate and initialize data structures
  clog << ">>> Initialize data structures" << std::endl;

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

  init_visibilities(visibilities, nr_baselines, nr_time, nr_channels, nr_polarizations);
  init_uvw(uvw, nr_stations, nr_baselines, nr_time, gridsize, subgridsize, w_planes);
  init_wavenumbers(wavenumbers, nr_channels);
  init_aterm(aterm, nr_stations, nr_polarizations, subgridsize);
  init_spheroidal(spheroidal, subgridsize);
  init_baselines(baselines, nr_stations, nr_baselines);
  init_grid(grid, gridsize, nr_polarizations);

  clog << endl;

  // Initialize interface to kernels
  clog << ">> Initialize proxy" << endl;
  //  idg::Compiler compiler = "/usr/bin/gcc";
  idg::Compiler compiler = "icpc";
  //  idg::Compilerflags compilerflags = "-Wall -O3 -g -DDEBUG -fopenmp -lfftw3 -lfftw3f";
  idg::Compilerflags compilerflags = "-Wall -O3 -g -DDEBUG -fopenmp -mkl";
  idg::proxy::SMP xeon(compiler, compilerflags, params, algparams);
  // Alternative: idg::proxy::SMP xeon("/usr/bin/gcc", "-O2 -fopenmp", params);

  // Run gridding
  xeon.grid_visibilities(visibilities, uvw, wavenumbers, aterm, 
  			 spheroidal, baselines, grid);

  // Run transform: Fourier->Image
  xeon.transform(idg::FourierDomainToImageDomain, grid);

  // do something 

  // Run transform: Image->Fourier
  xeon.transform(idg::ImageDomainToFourierDomain, grid);

  // Run degridding
  xeon.degrid_visibilities(grid, uvw, wavenumbers, aterm, 
			   spheroidal, baselines, visibilities);

  // // create another proxy:

  // // set compiler settings
  // idg::CompilerEnvironment cc; // default constructor reads from ENV
  // cc.set_c_compiler("icc");
  // cc.set_c_flags("-O3 -fopenmp");
  // cc.set_cpp_compiler("icpc");
  // cc.set_cpp_flags("-O2 -fopenmp");

  // idg::proxy::SMP another(cc, params);

  // another.grid_visibilities(dummy_ptr, dummy_ptr, dummy_ptr, dummy_ptr, 
  // 			    dummy_ptr, dummy_ptr, dummy_ptr);

  // another.transform(idg::FourierDomainToImageDomain, dummy_ptr);

  // // do something 

  // another.transform(idg::ImageDomainToFourierDomain, dummy_ptr);

  // another.degrid_visibilities(dummy_ptr, dummy_ptr, dummy_ptr, dummy_ptr, 
  // 			      dummy_ptr, dummy_ptr, dummy_ptr);


  // free memory for data structures?
  delete[] visibilities;
  delete[] uvw;
  delete[] wavenumbers;
  delete[] aterm;
  delete[] spheroidal;
  delete[] baselines;
  delete[] grid;


  // free(visibilities);
  // free(uvw);
  // free(wavenumbers);
  // free(aterm);
  // free(spheroidal);
  // free(baselines);
  // //  free(subgrids);
  // free(grid);
    
  return 0;
}
