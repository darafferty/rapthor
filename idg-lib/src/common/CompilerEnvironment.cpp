#include <iostream>
#include <iomanip>
#include "CompilerEnvironment.h"

using namespace std;

namespace idg {

  // auxiliary functions
  void CompilerEnvironment::print(ostream& os) const
  {
    const int fw1 = 10;
    const int fw2 = 30;

    os << "COMPILER ENVIRONMENT:" << endl;
    for (auto iter=value.begin(); iter!= value.end(); iter++)
      cout << setw(fw1) << left << iter->first << ":    "
	   << setw(fw2) << left << iter->second << endl;
  }


  void CompilerEnvironment::print() const
  {
    print(cout);
  }


  void CompilerEnvironment::read_parameters_from_env() 
  {
    const string DEFAULT_CC = "gcc";
    const string DEFAULT_CFLAGS = "-O2 -fopenmp";
    const string DEFAULT_CPP = "g++";
    const string DEFAULT_CPPFLAGS = "-O2 -fopenmp";
    const string DEFAULT_FC = "gfortran";
    const string DEFAULT_FFLAGS = "-O2 -fopenmp";
    const string DEFAULT_NVCC = "nvcc";
    const string DEFAULT_NVCCFLAGS = " ";
    // to be extended

    // CC
    char *cstr_cc = getenv(ENV_CC.c_str());
    if (cstr_cc != nullptr) {
      string cc = cstr_cc;
      value[CC] = cc;  
    } else {
      value[CC] = DEFAULT_CC;
    }

    // CFLAGS
    char *cstr_cflags = getenv(ENV_CFLAGS.c_str());
    if (cstr_cflags != nullptr) {
      string cflags = cstr_cflags;
      value[CFLAGS] = cflags;  
    } else {
      value[CFLAGS] = DEFAULT_CFLAGS;
    }

    // CPP
    char *cstr_cpp = getenv(ENV_CPP.c_str());
    if (cstr_cpp != nullptr) {
      string cpp = cstr_cpp;
      value[CPP] = cpp;  
    } else {
      value[CPP] = DEFAULT_CPP;
    }

    // CPPFLAGS
    char *cstr_cppflags = getenv(ENV_CPPFLAGS.c_str());
    if (cstr_cppflags != nullptr) {
      string cppflags = cstr_cppflags;
      value[CPPFLAGS] = cppflags;  
    } else {
      value[CPPFLAGS] = DEFAULT_CPPFLAGS;
    }

    // FC
    char *cstr_fc = getenv(ENV_FC.c_str());
    if (cstr_fc != nullptr) {
      string fc = cstr_fc;
      value[FC] = fc;  
    } else {
      value[FC] = DEFAULT_FC;
    }

    // FFLAGS
    char *cstr_fflags = getenv(ENV_FFLAGS.c_str());
    if (cstr_fflags != nullptr) {
      string fflags = cstr_fflags;
      value[FFLAGS] = fflags;  
    } else {
      value[FFLAGS] = DEFAULT_FFLAGS;
    }

    // NVCC
    char *cstr_nvcc = getenv(ENV_NVCC.c_str());
    if (cstr_nvcc != nullptr) {
      string nvcc = cstr_nvcc;
      value[NVCC] = nvcc;  
    } else {
      value[NVCC] = DEFAULT_NVCC;
    }

    // NVCCFLAGS
    char *cstr_nvccflags = getenv(ENV_NVCCFLAGS.c_str());
    if (cstr_nvccflags != nullptr) {
      string nvccflags = cstr_nvccflags;
      value[NVCCFLAGS] = nvccflags;  
    } else {
      value[NVCCFLAGS] = DEFAULT_NVCCFLAGS;
    }

    // to be extended

  } // read_parameters_from_env()


  // helper functions
  ostream& operator<<(ostream& os, const CompilerEnvironment& cc)
  {
    cc.print(os);
    return os;
  }


} // namespace idg
