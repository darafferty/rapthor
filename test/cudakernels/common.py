import os
import re
from ctypes import cdll
from cuda import cuda, nvrtc
import matplotlib.pyplot as plt
import numpy as np

import idg
import idg.util as util
from idg.data import Data


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

    return cuda_dir


def get_kernel_string(filename):
    # Helper function to recursively get a parent directory
    def get_parent_dir(dirname, level=1):
        if level == 0:
            return dirname
        else:
            parentdir = os.path.abspath(os.path.join(dirname, os.pardir))
            return get_parent_dir(parentdir, level - 1)

    # All the directories to look for kernel sources and header files
    prefixes = []
    dirname_kernel = os.path.dirname(filename)  # e.g. idg-lib/src/CUDA/common/kernels
    prefixes.append(dirname_kernel)
    dirname_src = get_parent_dir(dirname_kernel, 3)  # e.g. idg-lib/src
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
    filename_kernel = os.path.basename(filename)  # e.g. KernelGridder.cu
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
    (err,) = cuda.cuInit(0)
    cuda_check(err)

    # Retrieve handle for device 0
    err, device = cuda.cuDeviceGet(0)
    cuda_check(err)

    # Create context
    err, context = cuda.cuCtxCreate(0, device)
    cuda_check(err)

    return device, context


def get_cuda_target_architecture(device):
    err, major = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device
    )
    cuda_check(err)
    err, minor = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device
    )
    cuda_check(err)
    err, nvrtc_major, nvrtc_minor = nvrtc.nvrtcVersion()
    cuda_check(err)
    use_cubin = nvrtc_minor >= 1
    prefix = "sm" if use_cubin else "compute"
    return bytes(f"--gpu-architecture={prefix}_{major}{minor}", "ascii")


def compile_kernel(device, kernel_filename, kernel_name, compile_options=[]):
    # The kernel to compile
    root_dir = os.path.realpath(f"{__file__}/../../../..")
    root_dir = os.path.realpath(f"{__file__}/../../..")
    kernel_dir = f"{root_dir}/idg-lib/src/CUDA/common/kernels"
    kernel_path = f"{kernel_dir}/{kernel_filename}"
    kernel_string = get_kernel_string(kernel_path)

    # Create program
    err, prog = nvrtc.nvrtcCreateProgram(str.encode(kernel_string), b"", 0, [], [])
    cuda_check(err)

    # Get target architecture
    arch_arg = get_cuda_target_architecture(device)

    # Compile program
    compile_options = [
        bytes(compile_option, encoding="utf8") for compile_option in compile_options
    ]
    compile_options = [arch_arg] + compile_options
    compile_options.append(b"--use_fast_math")
    cuda_dir = get_cuda_dir()
    include_dir = f"{cuda_dir}/include"
    compile_options.append(bytes(f"--include-path={include_dir}", encoding="utf8"))
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(compile_options), compile_options)

    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        err, logSize = nvrtc.nvrtcGetProgramLogSize(prog)
        log = bytearray(logSize)
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError("Nvrtc Error: {}".format(log.decode()))

    # Get PTX from compilation
    err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
    cuda_check(err)
    ptx = b" " * ptxSize
    (err,) = nvrtc.nvrtcGetPTX(prog, ptx)
    cuda_check(err)

    # Load PTX as module data and retrieve function
    ptx = np.char.array(ptx)
    err, module = cuda.cuModuleLoadData(ptx.ctypes.data)
    cuda_check(err)
    err, kernel = cuda.cuModuleGetFunction(
        module, bytes(f"{kernel_name}", encoding="utf8")
    )
    cuda_check(err)

    return kernel


def cuda_mem_alloc(data):
    sizeof_data = np.array(data).nbytes
    err, d_data = cuda.cuMemAlloc(sizeof_data)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError("Cuda Error: {}".format(err))
    return sizeof_data, d_data


def cuda_memcpy_htod(d_data, h_data, bytes, stream):
    (err,) = cuda.cuMemcpyHtoDAsync(d_data, h_data, bytes, stream)
    cuda_check(err)


def cuda_memcpy_dtoh(d_data, h_data, bytes, stream):
    (err,) = cuda.cuMemcpyDtoHAsync(h_data, d_data, bytes, stream)
    cuda_check(err)


def cuda_stream_synchronize(stream):
    (err,) = cuda.cuStreamSynchronize(stream)
    cuda_check(err)


class DummyData:
    def __init__(self, device, stream, stokes_i_only = False):
        # IDG parameters
        self.grid_size = 2048
        self.nr_correlations = 4
        self.nr_polarizations = 4
        self.subgrid_size = 32
        self.image_size = 0.01
        self.nr_channels = 16
        self.nr_stations = 8
        self.nr_timeslots = 32
        self.nr_timesteps = 4096
        self.integration_time = 0.9
        layout_file = "LOFAR_lba.txt"

        # Stokes I only mode
        if (stokes_i_only):
            self.nr_correlations = 2
            self.nr_polarizations = 1

        # Derived IDG parameters
        self.nr_baselines = int((self.nr_stations * (self.nr_stations - 1)) / 2)

        # Initialize data
        data = Data(layout_file)

        # Limit baselines in length and number
        max_uv = data.compute_max_uv(self.grid_size, self.nr_channels)
        data.limit_max_baseline_length(max_uv)
        data.limit_nr_baselines(self.nr_baselines)
        self.data = data

        # Get remaining parameters
        self.image_size = data.compute_image_size(self.grid_size, self.nr_channels)

        # CUDA members
        self.device = device
        self.stream = stream

    def get_plan(self, uvw, frequencies):
        # Initialize baselines
        baselines = util.get_example_baselines(self.nr_stations, self.nr_baselines)

        # Initialize aterms offfsets
        aterm_offsets = util.get_example_aterm_offsets(
            self.nr_timeslots, self.nr_timesteps
        )

        # Create plan
        kernel_size = 9
        cell_size = self.image_size / self.grid_size
        plan = idg.Plan(
            kernel_size,
            self.subgrid_size,
            self.grid_size,
            cell_size,
            frequencies,
            uvw,
            baselines,
            aterm_offsets,
        )

        return plan

    def get_frequencies(self):
        frequencies = np.zeros(self.nr_channels, dtype=idg.frequenciestype)
        channel_offset = 0
        self.data.get_frequencies(
            frequencies, self.nr_channels, self.image_size, channel_offset
        )
        sizeof_frequencies, d_frequencies = cuda_mem_alloc(frequencies)
        cuda_memcpy_htod(d_frequencies, frequencies, sizeof_frequencies, self.stream)
        return frequencies, d_frequencies

    def get_wavenumbers(self, frequencies):
        # Initialize frequencies, wavenumbers
        speed_of_light = 299792458.0
        wavenumbers = np.array(
            2 * np.pi * frequencies / speed_of_light, dtype=frequencies.dtype
        )
        sizeof_wavenumbers, d_wavenumbers = cuda_mem_alloc(wavenumbers)
        cuda_memcpy_htod(d_wavenumbers, wavenumbers, sizeof_wavenumbers, self.stream)
        return wavenumbers, d_wavenumbers

    def get_uvw(self):
        # Initialize UVW coordinates
        uvw = np.zeros((self.nr_baselines, self.nr_timesteps, 3), dtype=np.float32)
        baseline_offset = 0
        time_offset = 0
        integration_time = 0.9
        self.data.get_uvw(
            uvw,
            self.nr_baselines,
            self.nr_timesteps,
            baseline_offset,
            time_offset,
            integration_time,
        )
        uvw[..., -1] = 0
        sizeof_uvw, d_uvw = cuda_mem_alloc(uvw)
        cuda_memcpy_htod(d_uvw, uvw, sizeof_uvw, self.stream)
        return uvw, d_uvw

    def get_visibilities(self, uvw, frequencies):
        # Initialize visibilities
        visibilities = util.get_example_visibilities(
            self.nr_baselines,
            self.nr_timesteps,
            self.nr_channels,
            self.nr_correlations,
            self.image_size,
            self.grid_size,
            uvw,
            frequencies,
        )
        sizeof_visibilities, d_visibilities = cuda_mem_alloc(visibilities)
        cuda_memcpy_htod(d_visibilities, visibilities, sizeof_visibilities, self.stream)
        return visibilities, d_visibilities

    def get_taper(self):
        # Initalize taper
        taper = util.get_identity_taper(self.subgrid_size)
        sizeof_taper, d_taper = cuda_mem_alloc(taper)
        cuda_memcpy_htod(d_taper, taper, sizeof_taper, self.stream)
        return taper, d_taper

    def get_aterms(self):
        # Initialize aterms
        aterms = util.get_example_aterms(
            self.nr_timeslots, self.nr_stations, self.subgrid_size, 4
        )
        sizeof_aterms, d_aterms = cuda_mem_alloc(aterms)
        cuda_memcpy_htod(d_aterms, aterms, sizeof_aterms, self.stream)
        return aterms, d_aterms

    def get_metadata(self, plan):
        # Initialize metadata
        nr_subgrids = plan.get_nr_subgrids()
        metadata = np.zeros(nr_subgrids, dtype=idg.metadatatype)
        plan.copy_metadata(metadata)
        sizeof_metadata, d_metadata = cuda_mem_alloc(metadata)
        cuda_memcpy_htod(d_metadata, metadata, sizeof_metadata, self.stream)
        return metadata, d_metadata

    def get_aterm_indices(self, plan):
        aterm_indices = np.zeros(
            (self.nr_baselines, self.nr_timesteps), dtype=np.int32
        )
        plan.copy_aterm_indices(aterm_indices)
        sizeof_aterm_indices, d_aterm_indices = cuda_mem_alloc(aterm_indices)
        cuda_memcpy_htod(
            d_aterm_indices, aterm_indices, sizeof_aterm_indices, self.stream
        )
        return aterm_indices, d_aterm_indices


def get_accuracy(a, b):
    diff = np.absolute(a) - np.absolute(b)
    return np.sum(np.power(diff.flatten(), 2)) / (np.max(a) * np.count_nonzero(a))


def compare_subgrids(s1, s2, subgrid_index=0, polarization_index=0):
    s1 = s1[subgrid_index, polarization_index, :, :]
    s2 = s2[subgrid_index, polarization_index, :, :]

    f, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(np.abs(s1))
    ax[0, 1].imshow(np.abs(s2))

    max = np.max(np.absolute(s1))
    s1 /= max
    s2 /= max

    diff = np.absolute(s2) - np.absolute(s1)
    scale = np.absolute(s2) / np.absolute(s1)

    ax[1, 0].imshow(diff)
    ax[1, 1].imshow(scale)

    ax[0, 0].set_title('subgrid 1')
    ax[0, 1].set_title('subgrid 2')
    ax[1, 0].set_title('diff')
    ax[1, 1].set_title('scale')

    plt.show()
