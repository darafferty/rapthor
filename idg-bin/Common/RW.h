#include <iostream>
#include <sstream>
#include <cstdlib>
#include <dlfcn.h>


#ifndef RUNTIME_WRAPPER
#define RUNTIME_WRAPPER


namespace rw {

	class Error : public std::exception {
		public:
			Error(const char* what):
			_what(what)
			{}
		
			const char *what() const throw();
		
		private:
			const char* _what;
	};

	class Source {
		public:
			Source();
			
			Source(const char *input_file_name);

			void compile(
				const char *compiler,
				const char *output_file_name,
				const char *compiler_options = 0);

		private:
			const char *input_file_name;
	};
	
	class Module {
		public:
			Module(const char *file_name) {
				_module = dlopen(file_name, RTLD_LAZY);
				if (!_module) {
					std::stringstream message;
					message << "Invalid module: " << file_name;
					throw Error(message.str().c_str());
				}
			}
			
			operator void*() {
				return _module;
			}

			~Module() {
				dlclose(_module);
			}

		private:
			void * _module;
	};

	class Function {
		public:
			Function(Module &module, const char *name) {
				_function = dlsym(module, name);
				if (!_function) {
					std::stringstream message;
					message << "Invalid function: " << name;
					throw Error(message.str().c_str());
				}
			}
			
			int* exec() {
				return ((int* (*)(void)) _function)();
			}
			
			void *get() {
				return _function;
			}

		private:
			void *_function;
	};
}

#endif
