import os
import re
from ctypes import cdll
from cuda import cuda, nvrtc
import numpy as np

def get_cuda_dir():
    # NVRTC is used, so we should be able to find libnvrtc.so
    handle = cdll.LoadLibrary("libnvrtc.so")

    # Calling a non-existent function will throw an AttributeError
    # with the path to the library
    library_path = ""
    try:
        handle.nonexistent()
    except AttributeError as e:
        library_path = str(e).split(":")[0]

    # Library directory: <prefix>/lib64
    library_dir = os.path.dirname(f"{library_path}")

    # CUDA directory: <prefix>/targets/<arch>-<os>
    cuda_dir = os.path.realpath(f"{library_dir}/..")
    print(cuda_dir)

    return cuda_dir


def get_kernel_string(filename):
    # Helper function to recursively get a parent directory
    def get_parent_dir(dirname, level=1):
        if (level == 0):
            return dirname
        else:
            parentdir = os.path.abspath(os.path.join(dirname, os.pardir))
            return get_parent_dir(parentdir, level -1)

    # All the directories to look for kernel sources and header files
    prefixes = []
    dirname_kernel = os.path.dirname(filename) # e.g. idg-lib/src/CUDA/common/kernels
    prefixes.append(dirname_kernel)
    dirname_src = get_parent_dir(dirname_kernel, 3) # e.g. idg-lib/src
    prefixes.append(dirname_src)

    # Helper function to recursively get file contents with local includes
    def add_file(filename, level=1):
        result = [""]
        for prefix in prefixes:
            try:
                with open(f"{prefix}/{filename}", "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        # Match lines where a local header file is included
                        if line.startswith('#include "'):
                            # Extract the name of the header file
                            m = re.findall(r'"(.*?)"', line)

                            # If a valid filename was found, add it recursively
                            if len(m):
                                header_file = m[0]
                                padding = "*" * level
                                result += f"/{padding} BEGIN INLINE {header_file} {padding}/\n"
                                result += add_file(header_file, level + 1)
                                result += "\n"
                                result += (
                                    f"/{padding} END INLINE {header_file} {padding}/\n"
                                )
                        else:
                            result += [line]
                break
            except FileNotFoundError:
                # It is ok if a file is not found, it might exists in another prefix
                pass
        return result

    # Start gathering all the source lines
    filename_kernel = os.path.basename(filename) # e.g. KernelGridder.cu
    source_lines = add_file(filename_kernel)

    # Return the result as a string
    return "".join(source_lines)

def cuda_check(err):
    name = cuda.cuGetErrorName(err)
    string = cuda.cuGetErrorString(err)
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cuda Error: {name}, {string}")
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"Nvrtc Error: {name}, {string}")
    else:
        raise RuntimeError(f"Unknown Error: {name}, {string}")


def cuda_initialize():
   # Initialize CUDA
    err, = cuda.cuInit(0)
    cuda_check(err)

    # Retrieve handle for device 0
    err, device = cuda.cuDeviceGet(0)
    cuda_check(err)

    # Create context
    err, context = cuda.cuCtxCreate(0, device)
    cuda_check(err)

    return device, context


def get_cuda_target_architecture(device):
    err, major = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)
    cuda_check(err)
    err, minor = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)
    cuda_check(err)
    err, nvrtc_major, nvrtc_minor = nvrtc.nvrtcVersion()
    cuda_check(err)
    use_cubin = (nvrtc_minor >= 1)
    prefix = 'sm' if use_cubin else 'compute'
    return bytes(f'--gpu-architecture={prefix}_{major}{minor}', 'ascii')


def compile_kernel(device, kernel_filename, kernel_name, compile_options=[]):
    # The kernel to compile
    root_dir = os.path.realpath(f"{__file__}/../../../..")
    kernel_dir = f"{root_dir}/idg-lib/src/CUDA/common/kernels"
    kernel_path = f"{kernel_dir}/{kernel_filename}"
    kernel_string = get_kernel_string(kernel_path)

    # Create program
    err, prog = nvrtc.nvrtcCreateProgram(str.encode(kernel_string), b"", 0, [], [])
    cuda_check(err)

    # Get target architecture
    arch_arg = get_cuda_target_architecture(device)

    # Compile program
    compile_options = [bytes(compile_option, encoding='utf8') for compile_option in compile_options]
    compile_options = [arch_arg] + compile_options
    #opts.append(b"-DBLOCK_SIZE_X=128")
    #opts.append(b"-DUNROLL_PIXELS=4")
    #opts.append(b"-DNUM_BLOCKS=4")
    #opts.append(b"-DUSE_EXTRAPOLATE=0")
    compile_options.append(b"--use_fast_math")
    cuda_dir = get_cuda_dir()
    include_dir = f"{cuda_dir}/include"
    compile_options.append(bytes(f"--include-path={include_dir}", encoding='utf8'))
    print(compile_options)
    err, = nvrtc.nvrtcCompileProgram(prog, len(compile_options), compile_options)

    if (err != nvrtc.nvrtcResult.NVRTC_SUCCESS):
        err, logSize = nvrtc.nvrtcGetProgramLogSize(prog)
        log = bytearray(logSize)
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError("Nvrtc Error: {}".format(log.decode()))

    # Get PTX from compilation
    err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
    cuda_check(err)
    ptx = b" " * ptxSize
    err, = nvrtc.nvrtcGetPTX(prog, ptx)
    cuda_check(err)

    # Load PTX as module data and retrieve function
    ptx = np.char.array(ptx)
    err, module = cuda.cuModuleLoadData(ptx.ctypes.data)
    cuda_check(err)
    err, kernel = cuda.cuModuleGetFunction(module, bytes(f"{kernel_name}", encoding='utf8'))
    cuda_check(err)

    return kernel

def cuda_mem_alloc(data):
    sizeof_data = np.array(data).nbytes
    err, d_data = cuda.cuMemAlloc(sizeof_data)
    if (err != cuda.CUresult.CUDA_SUCCESS):
        raise RuntimeError("Cuda Error: {}".format(err))
    return sizeof_data, d_data

def cuda_memcpy_htod(d_data, h_data, bytes, stream):
    err, = cuda.cuMemcpyHtoDAsync(d_data, h_data, bytes, stream)
    cuda_check(err)

def cuda_memcpy_dtoh(d_data, h_data, bytes, stream):
    err, = cuda.cuMemcpyDtoHAsync(h_data, d_data, bytes, stream)
    cuda_check(err)

def cuda_stream_synchronize(stream):
    err, = cuda.cuStreamSynchronize(stream)
    cuda_check(err)
