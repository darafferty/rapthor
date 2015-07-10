// This file mimics IDG-WRAPPER
// run with -DNR_STATIONS=x -DNR_CHANNELS=... or get with arguments...

#include <iostream>
#include "SMP.h"

using namespace std;

int main(int argc, char *argv[])
{
  // Set constants explicitly
  idg::Parameters params;    
  params.set_nr_stations(48);
  params.set_nr_timesteps(1024);
  params.set_nr_channels(256);
  params.set_nr_polarizations(4); 
  params.set_field_of_view(0.1);
  params.set_grid_size(4096);
  params.set_w_planes(1);

  // Print configuration
  clog << ">>> Configuration"  << std::endl;
  clog << params;

  // Initialize data structures
  clog << ">>> Initialize data structures" << std::endl;
  void *dummy_ptr;

  // Initialize interface to kernels
  clog << ">> Initialize proxy" << endl;
  idg::Compiler compiler = "/usr/bin/gcc";
  idg::Compilerflags compilerflags = "-Wall -O3 -g -DDEBUG -fopenmp -lfftw3 -lfftw3f";
  idg::proxy::SMP xeon(compiler, compilerflags, params);
  // Alternative: idg::proxy::SMP xeon("/usr/bin/gcc", "-O2 -fopenmp", params);

  // Run gridding
  xeon.grid_visibilities(dummy_ptr, dummy_ptr, dummy_ptr, dummy_ptr, 
			 dummy_ptr, dummy_ptr, dummy_ptr);

  // Run transform: Fourier->Image
  xeon.transform(idg::FourierDomainToImageDomain, dummy_ptr);

  // do something 

  // Run transform: Image->Fourier
  xeon.transform(idg::ImageDomainToFourierDomain, dummy_ptr);

  // Run degridding
  xeon.degrid_visibilities(dummy_ptr, dummy_ptr, dummy_ptr, dummy_ptr, 
			   dummy_ptr, dummy_ptr, dummy_ptr);


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

  return 0;
}
