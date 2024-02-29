#ifndef IDG_CUDA_GPU_UTILS_H_
#define IDG_CUDA_GPU_UTILS_H_

#include <map>
#include <set>
#include <string>

namespace cu {
class Context;
class Device;
class Module;
}  // namespace cu

namespace idg::kernel::cuda {
// Map for storing compile definitions that will be passed to the GPU kernel
// compiler on the command line. The key is the name of the preprocessor
// variable, a value should only be assigned if the preprocessor variable
// needs to have a value.
struct CompileDefinitions : std::map<std::string, std::string> {};

// Return default compile definitions.
CompileDefinitions defaultCompileDefinitions();

std::ostream& operator<<(std::ostream& os,
                         const CompileDefinitions& definitions);

// Set for storing compile flags that will be passed to the GPU runtime
// compiler. Flags generally don't have an associated value; if they do
// the value should become part of the flag in this set.
struct CompileFlags : std::set<std::string> {};

std::ostream& operator<<(std::ostream& os,
                         const idg::kernel::cuda::CompileFlags flags);

// Return default compile flags.
CompileFlags defaultCompileFlags();

// Compile \a source_filename and return the PTX code as string.
// \par source_filename Name of the file containing the source code to be
//      compiled to PTX.
// \par flags Set of flags that need to be passed to the compiler on the
//      command line.
// \par definitions Map of key/value pairs containing the preprocessor
//      variables that need to be passed to the compiler on the command
//      line.
// \par device Device to compile for.
std::string createPTX(const cu::Device& device, std::string& source_filename,
                      CompileDefinitions definitions = CompileDefinitions(),
                      CompileFlags flags = CompileFlags());

// Create a Module from a PTX (string).
// \par context The context that the Module should be associated with.
// \par source_filename Name of the
cu::Module createModule(const cu::Context& context,
                        const std::string& source_filename,
                        const std::string& ptx);

// Struct representing a CUDA Grid, which is similar to the @c dim3 type
// in the CUDA Runtime API.
struct Grid {
  Grid(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1);
  unsigned int x;
  unsigned int y;
  unsigned int z;
};

// Struct representing a CUDA Block, which is similar to the @c dim3 type
// in the CUDA Runtime API.
//
// @invariant x > 0, y > 0, z > 0
struct Block {
  Block(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1);
  unsigned int x;
  unsigned int y;
  unsigned int z;
};
}  // namespace idg::kernel::cuda

#endif
