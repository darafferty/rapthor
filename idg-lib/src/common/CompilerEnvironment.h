/** 
 *  \class CompilerEnvironment
 *
 *  \brief Collection of compiler settings 
 *
 *  Have a more detailed description here
 */

#ifndef IDG_COMPILERENV_H_
#define IDG_COMPILERENV_H_

#include <iostream>
#include <string>
#include <map>

#include "idg-config.h"

namespace idg {

  /// typedefs
  typedef std::string Compiler;
  typedef std::string Compilerflags;

  /// Define the environment names searched for in ENV
  static const std::string ENV_CC = "CC";
  static const std::string ENV_CFLAGS = "CFLAGS";
  static const std::string ENV_CPP = "CPP";
  static const std::string ENV_CPPFLAGS = "CPPFLAGS";
  static const std::string ENV_FC = "FC";
  static const std::string ENV_FFLAGS = "FFLAGS";
  static const std::string ENV_NVCC = "NVCC";
  static const std::string ENV_NVCCFLAGS = "NVCCLAGS";
  // extend to MPICC, ...

  /// Internal identifiers (keys) for various settings
  static const std::string CC = "CC";
  static const std::string CFLAGS = "CFLAGS";
  static const std::string CPP = "CPP";
  static const std::string CPPFLAGS = "CPPFLAGS";
  static const std::string FC = "FC";
  static const std::string FFLAGS = "FFLAGS";
  static const std::string NVCC = "NVCC";
  static const std::string NVCCFLAGS = "NVCCLAGS";
  // extend to MPICC, ...


  class CompilerEnvironment 
  {
  public:
    /// Create a CompilerEnvironment 
    /** The default constructor reads parameters 
      * from ENV using common CC, CFLAGS, CPP, CPPFLAGS, ... */
    CompilerEnvironment() {
      read_parameters_from_env();
    }

    // default copy constructor/assignment okay

    // default destructur
    ~CompilerEnvironment() = default;

    // set and get methods

    /** \brief Sets the C compiler.
      * \param s (alias or fullpath) C complier such that $s CFLAGS ... works */
    void set_c_compiler(std::string s) { value[CC] = s; }

    /** \brief Sets the C compiler flags.
      * \param s the C complier flags */
    void set_c_flags(std::string s) { value[CFLAGS] = s; }

    void set_cpp_compiler(std::string s) { value[CPP] = s; }
    void set_cpp_flags(std::string s) { value[CPPFLAGS] = s; }
    void set_f_compiler(std::string s) { value[FC] = s; }
    void set_f_flags(std::string s) { value[FFLAGS] = s; }
    void set_cu_compiler(std::string s) { value[NVCC] = s; }
    void set_cu_flags(std::string s) { value[NVCCFLAGS] = s; }

    const std::string& get_c_compiler() const { return value.at(CC); }
    const std::string& get_c_flags() const { return value.at(CFLAGS); }

    const std::string& get_cpp_compiler() const { return value.at(CPP); }
    const std::string& get_cpp_flags() const { return value.at(CPPFLAGS); }

    const std::string& get_f_compiler() const { return value.at(FC); }
    const std::string& get_f_flags() const { return value.at(FFLAGS); }

    const std::string& get_cu_compiler() const { return value.at(NVCC); }
    const std::string& get_cu_flags() const { return value.at(NVCCFLAGS); }

    // auxiliary functions
    void print() const;
    void print(std::ostream& os) const;
    void read_parameters_from_env();

    // data (is public so cc.value[CC] provides C compiler)
    std::map<std::string,std::string> value;
  }; 


  // helper functions
  std::ostream& operator<<(std::ostream& os, const CompilerEnvironment& cc);


} // namespace idg

#endif
