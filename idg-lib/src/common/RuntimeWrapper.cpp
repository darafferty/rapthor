#include <iostream>
#include <sstream>

#include "idg-config.h"
#include "RuntimeWrapper.h"

using namespace std;

namespace idg {

  namespace runtime {

    const char *Error::what() const throw() {
      return _what;
    }

    Source::Source(const char *input_file_name):
      input_file_name(input_file_name) {}

    void Source::compile(const char *compiler,
			 const char *output_file_name,
			 const char *compiler_options)
    {
        // Build command
        stringstream command_line;
        command_line << compiler;
        command_line << " -fPIC -shared";
        command_line << " -o ";
        command_line << output_file_name;
        command_line << ' ' << input_file_name;
        command_line << ' ' << compiler_options;

        #if defined(DEBUG) || defined(COMPILE_VERBOSE)
        #pragma omp critical(clog)
        {
            clog << "Compiling " << output_file_name << endl;
            clog << command_line.str() << endl;
        }
        #endif

        int retval = system(command_line.str().c_str());

        if (WEXITSTATUS(retval) != 0) {
            cerr << "Error compiling: " << input_file_name << endl;
            cerr << "retval: " << retval << endl;
            cerr << "WEXITSTATUS(retval)" << WEXITSTATUS(retval) << endl;
            exit(EXIT_FAILURE);
        }
    }

  } // namespace rw
} // namespace idg
