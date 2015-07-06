#include <iostream>
#include <sstream>
#include "RuntimeWrapper.h"

using namespace std;

namespace idg {

  namespace rw {
    
    const char *Error::what() const throw() {
      return _what;
    }
    
    Source::Source(const char *input_file_name):
      input_file_name(input_file_name) {}
    
    void Source::compile(const char *compiler,
			 const char *output_file_name,
			 const char *compiler_options)
    {
      
#pragma omp critical(clog)
      {
	clog << "Compiling " << output_file_name << endl;
	
	// Build command
	stringstream command_line;
	command_line << compiler;
	command_line << " -fPIC -shared -DRW ";
	command_line << compiler_options;
	command_line << " -o ";
	command_line << output_file_name;
	command_line << ' ' << input_file_name;
	
	clog << command_line.str() << endl;
	
	int retval = system(command_line.str().c_str());
	
	if (WEXITSTATUS(retval) != 0) {
	  cerr << "Error compiling: " << input_file_name << endl;
	  exit(EXIT_FAILURE);
	}
      }
    }
    
  } // namespace rw
  
} // namespace idg
