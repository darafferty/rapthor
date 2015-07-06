// This file mimics IDG-WRAPPER
// run with -DNR_STATIONS=x -DNR_CHANNELS=... or get with arguments...

#include <iostream>
#include "SMP.h"

using namespace std;

int main(int argc, char *argv[])
{
  // Set constants explicitly
  idg::CompileTimeConstants constants;    
  constants.set_nr_stations(48);
  constants.set_nr_timesteps(1024);
  constants.set_nr_channels(256);
  constants.set_nr_polarizations(4); 
  constants.set_field_of_view(0.1);
  constants.set_grid_size(4096);
  constants.set_subgrid_size(32);
  constants.set_chunk_size(128);
  constants.set_job_size(128);
  constants.set_w_planes(1);

  // Print configuration
  clog << ">>> Configuration"  << std::endl;
  clog << constants;

  // Initialize data structures
  clog << ">>> Initialize data structures" << std::endl;
  void *dummy_ptr;

  // Initialize interface to kernels
  clog << ">> Initialize proxy" << endl;
  idg::Compiler compiler = "/usr/bin/gcc";
  idg::Compilerflags compilerflags = "-Wall -O2 -g -DDEBUG -fopenmp";
  idg::proxy::SMP xeon(compiler, compilerflags, constants);
  // Alternative: idg::proxy::SMP xeon("/usr/bin/gcc", "-O2 -fopenmp", constants);

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

  // idg::proxy::SMP another(cc, constants);

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
