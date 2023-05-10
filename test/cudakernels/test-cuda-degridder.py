#!/usr/bin/env python

import ctypes
from cuda import cuda
import numpy as np
import pytest

from common import *


def compare_degridder(device, kernel1, kernel2, stokes_i_only):
    # Create stream
    err, stream = cuda.cuStreamCreate(0)

    # Initialize data
    data = DummyData(device, stream, stokes_i_only)

    nr_polarizations = data.nr_polarizations
    nr_correlations = data.nr_correlations
    subgrid_size = data.subgrid_size
    grid_size = data.grid_size
    image_size = data.image_size
    nr_channels = data.nr_channels
    nr_stations = data.nr_stations
    nr_baselines = data.nr_baselines
    nr_timesteps = data.nr_timesteps
    time_offset = 0
    w_step = 0
    shift_l = 0
    shift_m = 0

    uvw, d_uvw = data.get_uvw()
    frequencies, d_frequencies = data.get_frequencies()
    plan = data.get_plan(uvw, frequencies)
    wavenumbers, d_wavenumbers = data.get_wavenumbers(frequencies)
    taper, d_taper = data.get_taper()
    metadata, d_metadata = data.get_metadata(plan)
    aterms, d_aterms = data.get_aterms()
    aterm_indices, d_aterm_indices = data.get_aterm_indices(plan)

    # Debug
    # util.plot_metadata(metadata, uvw, frequencies, grid_size, subgrid_size, image_size)
    # plt.show()

    # Allocate subgrids
    nr_subgrids = plan.get_nr_subgrids()
    subgrids = np.random.random(
        (nr_subgrids, nr_polarizations, subgrid_size, subgrid_size)).astype(np.complex64)
    sizeof_subgrids, d_subgrids = cuda_mem_alloc(subgrids)
    cuda_memcpy_htod(d_subgrids, subgrids, sizeof_subgrids, stream)

    # Allocate visibilities
    visibilities1 = np.zeros(
        (nr_baselines, nr_timesteps, nr_channels, nr_correlations), dtype=np.complex64
    )

    sizeof_visibilities, d_visibilities = cuda_mem_alloc(visibilities1)
    cuda_memcpy_htod(d_visibilities, visibilities1, sizeof_visibilities, stream)

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
        d_taper,
        d_aterms,
        d_aterm_indices,
        d_metadata,
        d_subgrids,
    )

    arg_types = (
        ctypes.c_int,  # time_offset
        ctypes.c_int,  # nr_polarizations
        ctypes.c_int,  # grid_size
        ctypes.c_int,  # subgrid_size
        ctypes.c_float,  # image_size
        ctypes.c_float,  # w_step
        ctypes.c_float,  # shift_l
        ctypes.c_float,  # shift_m
        ctypes.c_int,  # nr_channels
        ctypes.c_int,  # nr_stations
        None,  # uvw
        None,  # wavenumbers
        None,  # visibilities
        None,  # taper
        None,  # aterms
        None,  # aterm_indices
        None,  # metadata
        None,  # subgrid
    )

    # Helper
    def launch_kernel(kernel):
        (err,) = cuda.cuLaunchKernel(
            kernel,
            nr_subgrids,  # grid x dim
            1,  # grid y dim
            1,  # grid z dim
            128,  # block x dim
            1,  # block y dim
            1,  # block z dim
            0,  # dynamic shared memory
            stream,  # stream
            (arg_values, arg_types),  # kernel arguments
            0,  # extra (ignore)
        )
        cuda_check(err)

    # Launch first kernel
    cuda_memcpy_htod(d_visibilities, visibilities1, sizeof_visibilities, stream)
    launch_kernel(kernel1)
    cuda_memcpy_dtoh(d_visibilities, visibilities1, sizeof_visibilities, stream)
    (err,) = cuda.cuStreamSynchronize(stream)
    cuda_check(err)

    # Run reference
    visibilities2 = np.zeros(visibilities1.shape, dtype=np.complex64)

    # Launch second kernel
    cuda_memcpy_htod(d_visibilities, visibilities2, sizeof_visibilities, stream)
    launch_kernel(kernel2)
    cuda_memcpy_dtoh(d_visibilities, visibilities2, sizeof_visibilities, stream)
    cuda_check(err)
    (err,) = cuda.cuStreamSynchronize(stream)

    # Debug
    # util.plot_visibilities(visibilities1 - visibilities2)
    # plt.show()

    # Check correctness
    tolerance = nr_baselines * nr_timesteps * nr_channels * np.finfo(np.float32).eps
    print(f"Tolerance: {tolerance}")
    error = get_accuracy(visibilities2, visibilities1)
    print(f"Error: {error}")
    assert abs(error) < tolerance
    max = np.max(np.absolute(visibilities2))
    assert np.allclose(visibilities2/max, visibilities2/max, atol=1e-5, rtol=1e-8)


@pytest.mark.parametrize("stokes_i_only", [False, True])
def test_degridder_default(stokes_i_only):
    print(f"test degridder default {'(Stokes I only)' if stokes_i_only else ''}")
    device, context = cuda_initialize()
    k1 = compile_kernel(device, "KernelDegridderReference.cu", "kernel_degridder")
    k2 = compile_kernel(device, "KernelDegridder.cu", "kernel_degridder")
    compare_degridder(device, k1, k2, stokes_i_only)

@pytest.mark.parametrize("stokes_i_only", [False, True])
def test_degridder_extrapolate(stokes_i_only):
    print(f"test degridder extrapolate {'(Stokes I only)' if stokes_i_only else ''}")
    device, context = cuda_initialize()
    k1 = compile_kernel(device, "KernelDegridderReference.cu", "kernel_degridder")
    k2 = compile_kernel(device, "KernelDegridder.cu", "kernel_degridder", ["-DUSE_EXTRAPOLATE"])
    compare_degridder(device, k1, k2, stokes_i_only)


if __name__ == "__main__":
    test_degridder_default(False)
    test_degridder_extrapolate(False)
