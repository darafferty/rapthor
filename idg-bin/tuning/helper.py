import argparse
import numpy as np
import os
import pynvml
import re

import kernel_tuner
from kernel_tuner.nvml import NVMLObserver


def get_default_parser():
    parser = argparse.ArgumentParser(description='Tune kernel.')
    parser.add_argument("--file", required=True, help="Path to kernel source file")
    parser.add_argument("--store-json", action="store_true")
    parser.add_argument("--tune-power-limit", action="store_true")
    parser.add_argument("--power-limit-steps", nargs="?")
    parser.add_argument("--tune-gr-clock", action="store_true")
    parser.add_argument("--gr-clock-steps", nargs="?")
    parser.add_argument("--tune-mem-clock", action="store_true")
    parser.add_argument("--mem-clock-steps", nargs="?")
    return parser


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


def get_supported_mem_clocks(dev, n=0):
    mem_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(dev)

    if n and len(mem_clocks) > n:
        mem_clocks = mem_clocks[::int(len(mem_clocks)/n)]

    return mem_clocks


def get_supported_gr_clocks(dev, mem_clock, n=0):
    assert mem_clock in get_supported_mem_clocks(dev)
    gr_clocks = pynvml.nvmlDeviceGetSupportedGraphicsClocks(
        dev, mem_clock
    )

    if n and (len(gr_clocks) > n):
        gr_clocks = gr_clocks[::int(len(gr_clocks)/n)]

    return gr_clocks


def setup_pwr_limit_tuning(dev, tune_params, n=None):
    print("> Setup power limit tuning")
    (
        power_limit_min,
        power_limit_max,
    ) = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(dev)
    power_limit_min *= 1e-3  # Convert to Watt
    power_limit_max *= 1e-3  # Convert to Watt
    power_limit_round = 5
    if n == None:
        n = int((power_limit_max - power_limit_min) / power_limit_round)
    tune_params["nvml_pwr_limit"] = power_limit_round * np.round(  # Rounded power limit values
        (np.linspace(power_limit_min, power_limit_max, n) / power_limit_round)
    )
    print(f"Tuning nvml_pwr_limit = {tune_params['nvml_pwr_limit']}")


def report_most_efficient(results, tune_params, metrics):
    best_config = min(results, key=lambda x: x["nvml_energy"])
    print("most efficient configuration:")
    kernel_tuner.util.print_config_output(
        tune_params, best_config, quiet=False, metrics=metrics, units=None)


def run_tuning(
    kernel_name,
    kernel_source,
    problem_size,
    kernel_arguments,
    tune_params,
    metrics,
    iterations,
    args
):
    tune_power_limit = args.tune_power_limit
    power_limit_steps = int(
        args.power_limit_steps) if args.power_limit_steps else None
    tune_gr_clock = args.tune_gr_clock
    gr_clock_steps = int(
        args.gr_clock_steps) if args.gr_clock_steps else None
    tune_mem_clock = args.tune_mem_clock
    mem_clock_steps = int(
        args.mem_clock_steps) if args.mem_clock_steps else None

    nvmlobserver = NVMLObserver(
        [
            "nvml_power",
            "nvml_energy",
            "core_freq",
            "mem_freq",
            "temperature",
        ]
    )
    dev = nvmlobserver.nvml.dev

    def tune_kernel():
        results, env = kernel_tuner.tune_kernel(
            kernel_name=kernel_name,
            kernel_source=kernel_source,
            problem_size=problem_size,
            grid_div_x=[],
            arguments=kernel_arguments,
            tune_params=tune_params,
            verbose=False,
            metrics=metrics,
            observers=[nvmlobserver],
            iterations=iterations,
            compiler_options=["-use_fast_math"],
            block_size_names=["BLOCK_SIZE_X"]
        )
        return results

    if tune_power_limit:
        setup_pwr_limit_tuning(dev, tune_params, power_limit_steps)

    if tune_mem_clock:
        # When tuning for both the memory clocks as well as for the graphics clock,
        # we need to run the tuner once for every memory clock, such that only the
        # valid combinations of the two will be tested.
        mem_clocks = get_supported_mem_clocks(dev, mem_clock_steps)

        results = []
        for mem_clock in mem_clocks:
            tune_params['nvml_mem_clock'] = [mem_clock]

            # Setup valid graphics clocks for the current memory clock
            gr_clocks = get_supported_gr_clocks(dev, mem_clock, gr_clock_steps)
            if not tune_gr_clock:
                gr_clocks = [max(gr_clocks)]
            tune_params['nvml_gr_clock'] = gr_clocks

            # Start tuning
            results_ = tune_kernel()

            report_most_efficient(results_, tune_params, metrics)
            results += results_

    else:
        # When not tuning for memory clock, we use the maximum supported memory
        # clock and select the corresponding supported graphics clocks.
        if tune_gr_clock:
            mem_clocks = get_supported_mem_clocks(dev)
            mem_clock = max(mem_clocks)
            gr_clocks = get_supported_gr_clocks(dev, mem_clock, gr_clock_steps)
            tune_params['nvml_gr_clock'] = gr_clocks

        # Start tuning
        results = tune_kernel()

        report_most_efficient(results, tune_params, metrics)

    return results
