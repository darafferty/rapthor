#!/usr/bin/env python

import ctypes
from cuda import cuda, nvrtc
import matplotlib.pyplot as plt
import numpy as np
import os

import idg
import idg.util as util
from idg.data import Data

import common_cuda

def ASSERT_DRV(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError('Cuda Error: {}'.format(err))
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError('Nvrtc Error: {}'.format(err))
    else:
        raise RuntimeError('Unknown error type: {}'.format(err))

def run():
    # The kernel to test
    root_dir = os.path.realpath(f"{__file__}/../../../..")
    kernel_dir = f"{root_dir}/idg-lib/src/CUDA/common/kernels"
    kernel_filename = "KernelGridder.cu"
    kernel_path = f"{kernel_dir}/{kernel_filename}"
    kernel_name = "kernel_gridder"
    kernel_string = common_cuda.get_kernel_string(kernel_path)
    with open("temp.cu", "w") as f:
        f.write(kernel_string)

    # Initialize CUDA
    err, = cuda.cuInit(0)
    ASSERT_DRV(err)

    # Retrieve handle for device 0
    err, cuDevice = cuda.cuDeviceGet(0)
    ASSERT_DRV(err)

    # Create program
    err, prog = nvrtc.nvrtcCreateProgram(str.encode(kernel_string), b"", 0, [], [])
    ASSERT_DRV(err)

    # Create context
    err, context = cuda.cuCtxCreate(0, cuDevice)
    ASSERT_DRV(err)

    # Get target architecture
    err, major = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice)
    ASSERT_DRV(err)
    err, minor = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice)
    ASSERT_DRV(err)
    err, nvrtc_major, nvrtc_minor = nvrtc.nvrtcVersion()
    ASSERT_DRV(err)
    use_cubin = (nvrtc_minor >= 1)
    prefix = 'sm' if use_cubin else 'compute'
    arch_arg = bytes(f'--gpu-architecture={prefix}_{major}{minor}', 'ascii')

    # Compile program
    opts = [arch_arg]
    #opts.append(b"-DBLOCK_SIZE_X=128")
    #opts.append(b"-DUNROLL_PIXELS=4")
    #opts.append(b"-DNUM_BLOCKS=4")
    #opts.append(b"-DUSE_EXTRAPOLATE=0")
    opts.append(b"--use_fast_math")
    cuda_dir = common_cuda.get_cuda_dir()
    include_dir = f"{cuda_dir}/include"
    opts.append(bytes(f"--include-path={include_dir}", encoding='utf8'))
    err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)

    if (err != nvrtc.nvrtcResult.NVRTC_SUCCESS):
        err, logSize = nvrtc.nvrtcGetProgramLogSize(prog)
        log = bytearray(logSize)
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError("Nvrtc Error: {}".format(log.decode()))

    # Get PTX from compilation
    err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
    ASSERT_DRV(err)
    ptx = b" " * ptxSize
    err, = nvrtc.nvrtcGetPTX(prog, ptx)
    ASSERT_DRV(err)

    # Load PTX as module data and retrieve function
    ptx = np.char.array(ptx)
    err, module = cuda.cuModuleLoadData(ptx.ctypes.data)
    ASSERT_DRV(err)
    err, kernel = cuda.cuModuleGetFunction(module, bytes(f"{kernel_name}", encoding='utf8'))
    ASSERT_DRV(err)

    # IDG parameters
    grid_size = 2048
    nr_correlations = 4
    nr_polarizations = 4
    subgrid_size = 32
    image_size = 0.01
    w_step = 0
    nr_channels = 16
    nr_stations = 3
    nr_timeslots = 32
    shift_l = 0.0
    shift_m = 0.0
    time_offset = 0
    nr_timesteps = 8192
    integration_time = 0.9
    layout_file = "LOFAR_lba.txt"
    kernel_size = 9

    # Derived IDG parameters
    nr_baselines = int((nr_stations * (nr_stations - 1)) / 2)

    # Initialize data
    data = Data(layout_file)

    # Limit baselines in length and number
    max_uv = data.compute_max_uv(grid_size, nr_channels)  # m
    data.limit_max_baseline_length(max_uv)
    data.limit_nr_baselines(nr_baselines)

    # Get remaining parameters
    image_size = data.compute_image_size(grid_size, nr_channels)
    cell_size = image_size / grid_size

    # Initialize data
    channel_offset = 0
    baseline_offset = 0
    time_offset = 0

    # Initialize frequencies, wavenumbers
    frequencies = np.zeros(nr_channels, dtype=idg.frequenciestype)
    data.get_frequencies(frequencies, nr_channels, image_size, channel_offset)
    speed_of_light = 299792458.0
    wavenumbers = np.array(2 * np.pi * frequencies / speed_of_light, dtype=frequencies.dtype)

    # Initialize UVW coordinates
    uvw = np.zeros((nr_baselines, nr_timesteps, 3), dtype=np.float32)
    data.get_uvw(
        uvw, nr_baselines, nr_timesteps, baseline_offset, time_offset, integration_time
    )
    uvw[...,-1] = 0

    # Initialize baselines
    baselines = util.get_example_baselines(nr_stations, nr_baselines)

    # Initialize aterms
    aterms = util.get_example_aterms(
        nr_timeslots, nr_stations, subgrid_size, nr_correlations
    )
    aterms_offsets = util.get_example_aterms_offset(nr_timeslots, nr_timesteps)

    # Initalize taper
    spheroidal = util.get_identity_spheroidal(subgrid_size)

    # Initialize visibilities
    visibilities = util.get_example_visibilities(
        nr_baselines,
        nr_timesteps,
        nr_channels,
        nr_correlations,
        image_size,
        grid_size,
        uvw,
        frequencies,
    )

    # Initialize metadata
    plan = idg.Plan(
        kernel_size,
        subgrid_size,
        grid_size,
        cell_size,
        frequencies,
        uvw,
        baselines,
        aterms_offsets)
    nr_subgrids = plan.get_nr_subgrids()
    metadata = np.zeros(nr_subgrids, dtype=idg.metadatatype)
    plan.copy_metadata(metadata)
    aterms_indices = np.zeros((nr_baselines, nr_timesteps), dtype=np.int32)
    plan.copy_aterms_indices(aterms_indices)

    # Allocate subgrids
    subgrids = np.zeros(
        (nr_subgrids, nr_polarizations, subgrid_size, subgrid_size), dtype=np.complex64)

    # Debug
    # util.plot_metadata(metadata, uvw, frequencies, grid_size, subgrid_size, image_size)
    # plt.show()

    # Create stream
    err, stream = cuda.cuStreamCreate(0)


    hX = np.random.rand(64).astype(dtype=np.float32)
    bufferSize = 64 * hX.itemsize
    err, dX = cuda.cuMemAlloc(bufferSize)
    ASSERT_DRV(err)
    err, = cuda.cuMemcpyHtoDAsync(dX, hX, bufferSize, stream)
    ASSERT_DRV(err)

    def cuda_mem_alloc(data):
        sizeof_data = np.array(data).nbytes
        err, d_data = cuda.cuMemAlloc(sizeof_data)
        if (err != cuda.CUresult.CUDA_SUCCESS):
            raise RuntimeError("Cuda Error: {}".format(err))
        return sizeof_data, d_data

    sizeof_uvw, d_uvw = cuda_mem_alloc(uvw)
    sizeof_wavenumbers, d_wavenumbers = cuda_mem_alloc(wavenumbers)
    sizeof_visbilities, d_visibilities = cuda_mem_alloc(visibilities)
    sizeof_spheroidal, d_spheroidal = cuda_mem_alloc(spheroidal)
    sizeof_aterms, d_aterms = cuda_mem_alloc(aterms)
    sizeof_aterms_indices, d_aterms_indices = cuda_mem_alloc(aterms_indices)
    sizeof_metadata, d_metadata = cuda_mem_alloc(metadata)
    sizeof_subgrids, d_subgrids = cuda_mem_alloc(subgrids)

    def cuda_memcpy_htod(d_data, h_data, bytes, stream):
        err, = cuda.cuMemcpyHtoDAsync(d_data, h_data, bytes, stream)
        ASSERT_DRV(err)

    def cuda_memcpy_dtoh(d_data, h_data, bytes, stream):
        err, = cuda.cuMemcpyDtoHAsync(h_data, d_data, bytes, stream)
        ASSERT_DRV(err)

    print(">>> Copy data from host to device")
    cuda_memcpy_htod(d_uvw, uvw, sizeof_uvw, stream)
    cuda_memcpy_htod(d_wavenumbers, wavenumbers, sizeof_wavenumbers, stream)
    cuda_memcpy_htod(d_visibilities, visibilities, sizeof_visbilities, stream)
    cuda_memcpy_htod(d_spheroidal, spheroidal, sizeof_spheroidal, stream)
    cuda_memcpy_htod(d_aterms, aterms, sizeof_aterms, stream)
    cuda_memcpy_htod(d_aterms_indices, aterms_indices, sizeof_aterms_indices, stream)
    cuda_memcpy_htod(d_metadata, metadata, sizeof_metadata, stream)
    cuda_memcpy_htod(d_subgrids, subgrids, sizeof_subgrids, stream)

    arg_values = (
        time_offset,
        nr_polarizations,
        grid_size,
        subgrid_size,
        image_size,
        w_step,
        shift_l,
        shift_m,
        nr_channels,
        nr_stations,
        d_uvw,
        d_wavenumbers,
        d_visibilities,
        d_spheroidal,
        d_aterms,
        d_aterms_indices,
        d_metadata,
        0, # average aterm is not used
        d_subgrids)

    arg_types = (
        ctypes.c_int, # time_offset
        ctypes.c_int, # nr_polarizations
        ctypes.c_int, # grid_size
        ctypes.c_int, # subgrid_size
        ctypes.c_float, # image_size
        ctypes.c_float, # w_step
        ctypes.c_float, # shift_l
        ctypes.c_float, # shift_m
        ctypes.c_int, # nr_channels
        ctypes.c_int, # nr_stations
        None, # uvw
        None, # wavenumbers
        None, # visibilities
        None, # spheroidal
        None, # aterms
        None, # aterms_indicies
        None, # metadata
        ctypes.c_int,# avg_aterm
        None  # subgrid
    )

    err, = cuda.cuStreamSynchronize(stream)
    ASSERT_DRV(err)

    # Launch kernel
    print(">>> Launch kernel")
    err, = cuda.cuLaunchKernel(
        kernel,
        nr_subgrids,  # grid x dim
        1,  # grid y dim
        1,  # grid z dim
        128,  # block x dim
        1,  # block y dim
        1,  # block z dim
        0,  # dynamic shared memory
        stream,  # stream
        (arg_values, arg_types), # kernel arguments
        0,  # extra (ignore)
    )
    ASSERT_DRV(err)

    err, = cuda.cuStreamSynchronize(stream)
    ASSERT_DRV(err)

    print(">>> Copy data from device to host")
    cuda_memcpy_dtoh(d_subgrids, subgrids, sizeof_subgrids, stream)

    err, = cuda.cuStreamSynchronize(stream)
    ASSERT_DRV(err)

if __name__ == "__main__":
    run()
