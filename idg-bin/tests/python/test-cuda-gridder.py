#!/usr/bin/env python

from re import A
from sre_constants import FAILURE
import common_cuda
import os
import numpy as np
import matplotlib.pyplot as plt

import idg
import idg.util as util
from idg.data import Data

from cuda import nvrtc
from cuda import cuda


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

    # Create program
    err, prog = nvrtc.nvrtcCreateProgram(str.encode(kernel_string), b"", 0, [], [])

    # Compile program
    opts = []
    #opts.append(b"-DBLOCK_SIZE_X=128")
    #opts.append(b"-DUNROLL_PIXELS=4")
    #opts.append(b"-DNUM_BLOCKS=4")
    #opts.append(b"-DUSE_EXTRAPOLATE=0")
    opts.append(b"--use_fast_math")
    opts.append(b"--gpu-architecture=compute_86")
    cuda_dir = common_cuda.get_cuda_dir()
    include_dir = f"{cuda_dir}/include"
    opts.append(bytes(f"--include-path={include_dir}", encoding='utf8'))
    err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)

    if (err != nvrtc.nvrtcResult.NVRTC_SUCCESS):
        err, logSize = nvrtc.nvrtcGetProgramLogSize(prog)
        programLog = bytearray(logSize)
        nvrtc.nvrtcGetProgramLog(prog, programLog)
        print(programLog.decode())
        exit(FAILURE)

    # Get PTX from compilation
    err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
    ptx = b" " * ptxSize
    err, = nvrtc.nvrtcGetPTX(prog, ptx)

    # Initialize CUDA Driver API
    err, = cuda.cuInit(0)

    # Retrieve handle for device 0
    err, cuDevice = cuda.cuDeviceGet(0)

    # Create context
    err, context = cuda.cuCtxCreate(0, cuDevice)

    # Load PTX as module data and retrieve function
    ptx = np.char.array(ptx)
    err, module = cuda.cuModuleLoadData(ptx.ctypes.data)
    err, kernel = cuda.cuModuleGetFunction(module, bytes(f"{kernel_name}", encoding='utf8'))
    print(kernel)

    # IDG parameters
    grid_size = 1024
    nr_correlations = 4
    nr_polarizations = 4
    subgrid_size = 32
    image_size = 0.01
    w_step = 0
    nr_channels = 16
    nr_stations = 10
    nr_timeslots = 32
    shift_l = 0.0
    shift_m = 0.0
    time_offset = 0
    nr_timesteps = 256
    integration_time = 0.9
    layout_file = "SKA1_low.txt"
    kernel_size = 9

    # Derived IDG parameters
    nr_baselines = int((nr_stations * (nr_stations - 1)) / 2)
    nr_subgrids = nr_baselines * nr_timeslots

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

    uvw = np.zeros((nr_baselines, nr_timesteps, 3), dtype=np.float32)
    frequencies = np.zeros(nr_channels, dtype=idg.frequenciestype)
    data.get_frequencies(frequencies, nr_channels, image_size, channel_offset)
    data.get_uvw(
        uvw, nr_baselines, nr_timesteps, baseline_offset, time_offset, integration_time
    )

    baselines = util.get_example_baselines(nr_stations, nr_baselines)
    aterms = util.get_example_aterms(
        nr_timeslots, nr_stations, subgrid_size, nr_correlations
    )
    aterms_offsets = util.get_example_aterms_offset(nr_timeslots, nr_timesteps)
    spheroidal = util.get_identity_spheroidal(subgrid_size)
    shift = np.zeros(2, dtype=np.float32)

    # set w to zero
    uvw[...,-1] = 0

    # Initialize dummy visibilities
    example_visibilities = util.get_example_visibilities(
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
    util.plot_metadata(metadata, uvw, frequencies, grid_size, subgrid_size, image_size)
    plt.show()


if __name__ == "__main__":
    run()
