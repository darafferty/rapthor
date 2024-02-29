#include <boost/format.hpp>
#include <cudawrappers/cu.hpp>
#include <cudawrappers/nvrtc.hpp>

#include "idg-common.h"
#include "gpu_utils.h"

namespace {

// Return the highest compute target supported by the given device
CUjit_target computeTarget(const cu::Device& device) {
  unsigned major =
      device.getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR>();
  unsigned minor =
      device.getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR>();

  return (CUjit_target)(major * 10 + minor);
}

// Translate a compute target to a virtual architecture (= the version
// the .cu file is written in).
std::string get_virtarch(CUjit_target target) {
  return str(boost::format("compute_%d") % target);
}

std::string prefixPath() { return idg::auxiliary::get_lib_dir() + "/idg-cuda"; }
}  // namespace

namespace idg::kernel::cuda {
std::ostream& operator<<(std::ostream& os,
                         const CompileDefinitions& definitions) {
  for (const std::pair<std::string, std::string>& definition : definitions) {
    os << " -D" << definition.first;
    if (!definition.second.empty()) {
      os << "=" << definition.second;
    }
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const CompileFlags flags) {
  for (const std::string& flag : flags) {
    os << " " << flag;
  }
  return os;
}

CompileDefinitions defaultCompileDefinitions() {
  CompileDefinitions definitions;
  return definitions;
}

CompileFlags defaultCompileFlags() {
  CompileFlags flags;
  flags.insert("--restrict");
  flags.insert("-lineinfo");
  flags.insert("--use_fast_math");
  return flags;
}

std::string createPTX(const cu::Device& device, std::string& source_filename,
                      CompileDefinitions definitions, CompileFlags flags) {
  flags.insert(str(boost::format("--gpu-architecture=%s") %
                   get_virtarch(computeTarget(device))));

  // Add default definitions and flags
  CompileDefinitions defaultDefinitions(defaultCompileDefinitions());
  definitions.insert(defaultDefinitions.begin(), defaultDefinitions.end());

  CompileFlags defaultFlags(defaultCompileFlags());
  flags.insert(defaultFlags.begin(), defaultFlags.end());

  // Prefix the CUDA kernel filename if it's a relative path.
  if (!source_filename.empty() && source_filename[0] != '/') {
    source_filename = prefixPath() + "/" + source_filename;
  }

#if defined(DEBUG)
  std::clog << "Starting runtime compilation:\n\t" << source_filename << flags
            << definitions << std::endl;
#endif

  // Append compile flags and compile definitions to get compile options
  std::stringstream options_stream;
  options_stream << flags;
  options_stream << definitions;
  options_stream << " -rdc=true";
  options_stream << " -I" << CUDA_TOOLKIT_INCLUDE_DIRECTORIES;

  // Create vector of compile options with strings
  std::vector<std::string> options_vector{
      std::istream_iterator<std::string>{options_stream},
      std::istream_iterator<std::string>{}};

  // Create array of compile options with character arrays
  std::vector<const char*> options_c_char;
  std::transform(options_vector.begin(), options_vector.end(),
                 std::back_inserter(options_c_char),
                 [](const std::string& option) { return option.c_str(); });

  // Open source file
  std::ifstream input_stream;
  input_stream.open(source_filename, std::ifstream::in);
  if (input_stream.fail()) {
    std::stringstream message;
    message << "Failed to open file:\n\t" << source_filename;
    throw std::runtime_error(message.str());
  }
  std::ostringstream file_contents;
  file_contents << input_stream.rdbuf();
  input_stream.close();

  // Try to compile source into NVRTC program
  nvrtc::Program program(file_contents.str(), source_filename.c_str());
  try {
    program.compile(options_vector);
  } catch (nvrtc::Error& error) {
    // Print compilation log (if any)
    std::string log = program.getLog();
    if (log.size())  // ignore logs with only trailing NULL
    {
      std::clog << log << std::endl;
    }

    throw error;
  }

  return program.getPTX();
}

cu::Module createModule(const cu::Context& context,
                        const std::string& source_filename,
                        const std::string& ptx) {
  const unsigned int BUILD_MAX_LOG_SIZE = 4095;

  cu::Module::optionmap_t options;

  size_t infoLogSize = BUILD_MAX_LOG_SIZE + 1;
  size_t errorLogSize = BUILD_MAX_LOG_SIZE + 1;

  std::vector<char> infoLog(infoLogSize);
  options[CU_JIT_INFO_LOG_BUFFER] = &infoLog[0];
  options[CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES] =
      reinterpret_cast<void*>(infoLogSize);

  std::vector<char> errorLog(errorLogSize);
  options[CU_JIT_ERROR_LOG_BUFFER] = &errorLog[0];
  options[CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES] =
      reinterpret_cast<void*>(errorLogSize);

  try {
    cu::Module module(reinterpret_cast<const void*>(ptx.data()), options);
    // guard against bogus output
    if (infoLogSize > infoLog.size()) {
      infoLogSize = infoLog.size();
    }
    infoLog[infoLogSize - 1] = '\0';
    std::clog << "Build info for '" << source_filename << &infoLog[0]
              << std::endl;
    return module;
  } catch (std::runtime_error& exc) {
    // guard against bogus output
    if (errorLogSize > errorLog.size()) {
      errorLogSize = errorLog.size();
    }
    errorLog[errorLogSize - 1] = '\0';
    std::cerr << "Build errors for '" << source_filename << &errorLog[0]
              << std::endl;
    throw;
  }
}

Grid::Grid(unsigned int x_, unsigned int y_, unsigned int z_)
    : x(x_), y(y_), z(z_) {}

Block::Block(unsigned int x_, unsigned int y_, unsigned int z_)
    : x(x_), y(y_), z(z_) {}

}  // namespace idg::kernel::cuda
