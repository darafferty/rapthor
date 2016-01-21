#include "CU.h"

#include <iostream>
#include <sstream>


namespace cu {

const char *Error::what() const throw() {
	switch (_result) {
		case CUDA_SUCCESS:
			return "success";
		case CUDA_ERROR_INVALID_VALUE:
			return "invalid value";
		case CUDA_ERROR_OUT_OF_MEMORY:
			return "out of memory";
		case CUDA_ERROR_NOT_INITIALIZED:
			return "not initialized";
		case CUDA_ERROR_DEINITIALIZED:
			return "deinitialized";
		case CUDA_ERROR_PROFILER_DISABLED:
			return "profiler disabled";
		case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
			return "profiler not initialized";
		case CUDA_ERROR_PROFILER_ALREADY_STARTED:
			return "profiler already started";
		case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
			return "profiler already stopped";
		case CUDA_ERROR_NO_DEVICE:
			return "no device";
		case CUDA_ERROR_INVALID_DEVICE:
			return "invalid device";
		case CUDA_ERROR_INVALID_IMAGE:
			return "invalid image";
		case CUDA_ERROR_INVALID_CONTEXT:
			return "invalid context";
		case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
			return "context already current";
		case CUDA_ERROR_MAP_FAILED:
			return "map failed";
		case CUDA_ERROR_UNMAP_FAILED:
			return "unmap failed";
		case CUDA_ERROR_ARRAY_IS_MAPPED:
			return "array is mapped";
		case CUDA_ERROR_ALREADY_MAPPED:
			return "already mapped";
		case CUDA_ERROR_NO_BINARY_FOR_GPU:
			return "no binary for GPU";
		case CUDA_ERROR_ALREADY_ACQUIRED:
			return "already acquired";
		case CUDA_ERROR_NOT_MAPPED:
			return "not mapped";
		case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
			return "not mapped as array";
		case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
			return "not mapped as pointer";
		case CUDA_ERROR_ECC_UNCORRECTABLE:
			return "ECC uncorrectable";
		case CUDA_ERROR_UNSUPPORTED_LIMIT:
			return "unsupported limit";
		case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
			return "context already in use";
		case CUDA_ERROR_INVALID_SOURCE:
			return "invalid source";
		case CUDA_ERROR_FILE_NOT_FOUND:
			return "file not found";
		case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
			return "shared object symbol not found";
		case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
			return "shared object init failed";
		case CUDA_ERROR_OPERATING_SYSTEM:
			return "operating system";
		case CUDA_ERROR_INVALID_HANDLE:
			return "invalid handle";
		case CUDA_ERROR_NOT_FOUND:
			return "not found";
		case CUDA_ERROR_NOT_READY:
			return "not ready";
		case CUDA_ERROR_LAUNCH_FAILED:
			return "launch failed";
		case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
			return "launch out of resources";
		case CUDA_ERROR_LAUNCH_TIMEOUT:
			return "launch timeout";
		case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
			return "launch incompatible texturing";
		case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
			return "peer access already enabled";
		case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
			return "peer access not enabled";
		case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
			return "primary context active";
		case CUDA_ERROR_CONTEXT_IS_DESTROYED:
			return "context is destroyed";
		case CUDA_ERROR_UNKNOWN:
			return "unknown";
	default:
			return "unknown error code";
	}
}


Source::Source(const char *input_file_name):
	input_file_name(input_file_name)
	{}


void Source::compile(const char *output_file_name, const char *compiler_options) {
    std::clog << "Compiling " << output_file_name << std::endl;
    
	std::stringstream command_line;
	command_line << "nvcc -ptx ";
	command_line << compiler_options;
	command_line << " -o ";
	command_line << output_file_name;
	command_line << ' ' << input_file_name;

    #if defined(DEBUG)
    std::clog << command_line.str() << std::endl;
    #endif
    int retval = system(command_line.str().c_str());
	    
    if (WEXITSTATUS(retval) != 0) {
    	throw cu::Error(CUDA_ERROR_INVALID_SOURCE);
	}
}

}
