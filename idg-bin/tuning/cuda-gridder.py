#!/usr/bin/env python

import json
import numpy as np
import os
from collections import OrderedDict

import helper
import idg


def tune():
    # Parse command line arguments
    parser = helper.get_default_parser()
    parser.add_argument("--block-size", nargs="+", default=[32 * i for i in range(1,9)])
    parser.add_argument("--num-blocks", nargs="+", default=range(9))
    parser.add_argument("--unroll-pixels", nargs="+", default=[1, 2, 4])
    args = parser.parse_args()

    # The kernel to tune
    kernel_name = "kernel_gridder"
    kernel_string = helper.get_kernel_string(args.file)
    kernel_source = f"{kernel_name}.cu"
    with open(kernel_source, "w") as f:
        f.write(kernel_string)

    # Tuning parameters
    tune_params = OrderedDict()
    tune_params["BLOCK_SIZE_X"] = args.block_size
    tune_params["UNROLL_PIXELS"] = args.unroll_pixels
    tune_params["NUM_BLOCKS"] = args.num_blocks
    tune_params["USE_EXTRAPOLATE"] = [0, 1]

    # IDG parameters
    grid_size = 8192
    nr_correlations = 4
    nr_polarizations = 4
    subgrid_size = 32
    image_size = 0.01
    w_step = 0
    nr_channels = 16
    nr_stations = 30
    nr_timeslots = 64
    shift_l = 0.0
    shift_m = 0.0
    time_offset = 0
    nr_timesteps_per_subgrid = 64

    # Derived IDG parameters
    nr_baselines = int((nr_stations * (nr_stations - 1)) / 2)
    nr_subgrids = nr_baselines * nr_timeslots
    nr_timesteps_all_baselines = nr_subgrids * nr_timesteps_per_subgrid
    nr_timesteps_per_baseline = int(nr_timesteps_all_baselines / nr_baselines)
    nr_aterms = nr_timeslots * nr_baselines

    # Amount of work performed
    mvis = 1e-6 * nr_timesteps_all_baselines * nr_channels
    gflops = 1e-9 * idg.flops_gridder(
        nr_channels,
        nr_timesteps_all_baselines,
        nr_subgrids,
        subgrid_size,
        nr_correlations
    )

    # Initialize metadata
    metadata = idg.get_metadata(
        nr_subgrids, nr_baselines, nr_timeslots, nr_timesteps_per_subgrid, nr_channels)

    # Initialize aterms indices
    aterm_indices = idg.get_aterm_indices(
        nr_timesteps_per_baseline, nr_timesteps_per_subgrid)

    # The following data types should have the correct dimensions,
    # but the contents are not relevant for performance
    np.random.seed(0)
    uvw = np.random.randn(nr_baselines, nr_timesteps_per_baseline, 3).astype(
        dtype=np.float32
    )
    wavenumbers = np.random.randn(nr_channels).astype(dtype=np.float32)
    visibilities = np.random.randn(
        nr_baselines, nr_timesteps_per_baseline, nr_channels, nr_correlations
    ).astype(dtype=np.complex64)
    taper = np.random.randn(
        subgrid_size, subgrid_size).astype(dtype=np.float32)
    aterms = np.random.randn(nr_aterms, subgrid_size, subgrid_size, 4).astype(
        dtype=np.complex64
    )
    subgrids = np.random.randn(
        nr_subgrids, nr_correlations, subgrid_size, subgrid_size
    ).astype(dtype=np.complex64)
    avg_aterm = np.random.randn(4, 4).astype(dtype=np.complex64)

    # Kernel arguments
    kernel_arguments = [
        np.int32(time_offset),
        np.int32(nr_polarizations),
        np.int32(grid_size),
        np.int32(subgrid_size),
        np.float32(image_size),
        np.float32(w_step),
        np.float32(shift_l),
        np.float32(shift_m),
        np.int32(nr_channels),
        np.int32(nr_stations),
        uvw,
        wavenumbers,
        visibilities,
        taper,
        aterms,
        aterm_indices,
        metadata,
        avg_aterm,
        subgrids,
    ]

    metrics = OrderedDict()
    metrics["J"] = lambda p: p["nvml_energy"]
    metrics["W"] = lambda p: p["nvml_power"]
    metrics["GFLOPS"] = lambda p: 1e3 * gflops / p["time"]
    metrics["GFLOPS/W"] = lambda p: gflops / p["nvml_energy"]
    metrics["MVIS/S"] = lambda p: 1e3 * mvis / p["time"]

    results = helper.run_tuning(kernel_name=kernel_name,
                                kernel_source=kernel_source,
                                problem_size=nr_subgrids,
                                kernel_arguments=kernel_arguments,
                                tune_params=tune_params,
                                metrics=metrics,
                                iterations=5,
                                args=args)

    if args.store_json:
        with open(f"cuda-gridder-{os.getpid()}.json", "w") as fp:
            json.dump(results, fp)


if __name__ == "__main__":
    tune()
