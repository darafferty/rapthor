#!/usr/bin/env python

import ctypes
from cuda import cuda
import numpy as np

from common_cuda import *


def compare_gridder(device, kernel1, kernel2):
    # Create stream
    err, stream = cuda.cuStreamCreate(0)

    # Initialize data
    data = TestData(device, stream)

    nr_polarizations = data.nr_polarizations
    subgrid_size = data.subgrid_size
    grid_size = data.grid_size
    image_size = data.image_size
    nr_channels = data.nr_channels
    nr_stations = data.nr_stations
    time_offset = 0
    w_step = 0
    shift_l = 0
    shift_m = 0

    uvw, d_uvw = data.get_uvw()
    frequencies, d_frequencies = data.get_frequencies()
    plan = data.get_plan(uvw, frequencies)
    wavenumbers, d_wavenumbers = data.get_wavenumbers(frequencies)
    visibilities, d_visibilities = data.get_visibilities(uvw, frequencies)
    taper, d_taper = data.get_taper()
    metadata, d_metadata = data.get_metadata(plan)
    aterms, d_aterms = data.get_aterms()
    aterms_indices, d_aterms_indices = data.get_aterms_indices(plan)

    # Debug
    # util.plot_metadata(metadata, uvw, frequencies, grid_size, subgrid_size, image_size)
    # plt.show()

    # Allocate subgrids
    nr_subgrids = plan.get_nr_subgrids()
    subgrids = np.zeros(
        (nr_subgrids, nr_polarizations, subgrid_size, subgrid_size), dtype=np.complex64
    )
    sizeof_subgrids, d_subgrids = cuda_mem_alloc(subgrids)
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
        d_taper,
        d_aterms,
        d_aterms_indices,
        d_metadata,
        0,  # average aterm is not used
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
        None,  # spheroidal
        None,  # aterms
        None,  # aterms_indicies
        None,  # metadata
        ctypes.c_int,  # avg_aterm
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
    launch_kernel(kernel1)
    cuda_memcpy_dtoh(d_subgrids, subgrids, sizeof_subgrids, stream)
    (err,) = cuda.cuStreamSynchronize(stream)
    cuda_check(err)

    # Run reference
    subgrids_reference = np.zeros(
        (nr_subgrids, nr_polarizations, subgrid_size, subgrid_size), dtype=np.complex64
    )

    # Launch second kernel
    launch_kernel(kernel2)
    cuda_memcpy_dtoh(d_subgrids, subgrids_reference, sizeof_subgrids, stream)
    cuda_check(err)
    (err,) = cuda.cuStreamSynchronize(stream)

    # Debug
    # compare_subgrids(subgrids_reference, subgrids)

    # Check correctness
    print(f"Error: {get_accuracy(subgrids_reference, subgrids)}")

    max = np.max(np.absolute(subgrids_reference))
    assert np.allclose(subgrids_reference/max, subgrids/max, atol=1e-5, rtol=1e-6)


def test_gridder_default():
    device, context = cuda_initialize()
    k1 = compile_kernel(device, "KernelGridderReference.cu", "kernel_gridder")
    k2 = compile_kernel(device, "KernelGridder.cu", "kernel_gridder")
    compare_gridder(device, k1, k2)

def test_gridder_extrapolate():
    device, context = cuda_initialize()
    k1 = compile_kernel(device, "KernelGridderReference.cu", "kernel_gridder")
    k2 = compile_kernel(device, "KernelGridder.cu", "kernel_gridder", ["-DUSE_EXTRAPOLATE"])
    compare_gridder(device, k1, k2)


if __name__ == "__main__":
    test_gridder_default()
    test_gridder_extrapolate()
