#include "RW.h"

namespace rw {

	const char *Error::what() const throw() {
		return _what;
	}

	Source::Source(const char *input_file_name):
	  input_file_name(input_file_name) {}
	
	void Source::compile(
		const char *compiler,
		const char *output_file_name,
		const char *compiler_options)
		{
		
		#pragma omp critical(clog)
		{
		    std::clog << "Compiling " << output_file_name << std::endl;

		    // Build command
		    std::stringstream command_line;
		    command_line << compiler;
		    command_line << " -fPIC -shared -DRW ";
		    command_line << compiler_options;
		    command_line << " -o ";
		    command_line << output_file_name;
		    command_line << ' ' << input_file_name;

		    std::clog << command_line.str() << std::endl;

		    int retval = system(command_line.str().c_str());
        
		    if (WEXITSTATUS(retval) != 0) {
                std::cerr << "Error compiling: " << input_file_name << std::endl;
                exit(EXIT_FAILURE);
		    }
	    }
	}

}
