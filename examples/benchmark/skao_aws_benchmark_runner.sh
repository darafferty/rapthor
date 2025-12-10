#!/bin/bash

###############################################################################
# Rapthor Benchmark Script for SKAO AWS cluster
#
# This script is intended to be run by the `run_all_tests.sh` script in
# ska-sdp-spack/scripts/:
# https://gitlab.com/ska-telescope/sdp/ska-sdp-spack/-/blob/sp-5859-add-bench-scripts/scripts/run_all_tests.sh
# The wrapper script loads the environmental variables from the file
# `ska-sdp-spack/scripts/test_specs/_ical.sh`, and executes this script by
# running the benchmark wrapper script `run_test.sh`.

# Environment variables:
# ---------------------------------------------------------------------------- #
# The `run_all_tests.sh` wrapper script will set the 
# META_MODULE variable to the latest release before executing this script.
#
# The following path variables are set in `run_test.sh`:
#     INPUT_PATH   : Path containing input Measurement Set (*.ms),
#                    pipeline config file, and any extra inputs.
#     WORK_PATH    : Scratch/work directory
#     OUTPUT_PATH  : Directory where pipeline outputs will be written
#     REPORT_PATH  : Directory where reports will be copied
#     CODE_PATH    : Path to the rapthor code repository
#     LOG_PATH     : Directory where logs will be written
#
# The following are set by `_ical.sh`:
#     PARTITION    : SLURM partition to use
#     NODE_COUNT   : Number of nodes to use
# 
# The following environmental variables should be set to configure compute
# resources for rapthor. These are used to populate the `[cluster]` section of
# the template parset file.:
#    RAPTHOR_BATCH_SYSTEM
#    RAPTHOR_MAX_NODES
#    RAPTHOR_CPUS_PER_TASK
#    RAPTHOR_MEM_PER_NODE_GB
#    RAPTHOR_MAX_CORES
#    RAPTHOR_MAX_THREADS
#    RAPTHOR_DECONV_THREADS
#    RAPTHOR_PARALLEL_GRIDDING_THREADS
#
# Parset and Strategy:
# ---------------------------------------------------------------------------- #
# The `run_test.sh` wrapper script will synchronise the parset and strategy
# files into the INPUT_PATH before executing this script. The parset used for
# this run is available in the rapthor repository under
# `/examples/rapthor_skao_benchmark_template.parset`. 
# These files are also hosted on S3 at: 
#   s3://skao-sdp-testdata/gecko/ical-benchmark-input/extra-inputs/2025.10.3
# 
# Since the parset needs to refer to paths within the benchmark environment,
# which is set by the wrapper script, we need to update paths in the template
# parset mentioned above.
#
# Dataset:
# ---------------------------------------------------------------------------- #
# The benchmark uses a simulated AA2 with 68 stations dataset generated with
# OSKAR. Details are available on confluence:
# https://confluence.skatelescope.org/display/SE/%5BSimulations%5D+--+New+data+for+benchmarking
# 
# The data is available on S3 at:
# s3://skao-sdp-testdata/PI28-Low-G3/product/eb-low68s_tec_lotss_noise_small-20000104-00000/ska-sdp/pb-low68s_tec_lotss_noise_small-20000104-00000/visibility.scan-400_applybeam.ms
# Also available on the shared FSx storage at:
# /shared/fsx1/shared/product/eb-low68s_tec_lotss_noise_small-20000104-00000/ska-sdp/pb-low68s_tec_lotss_noise_small-20000104-00000/visibility.scan-400_applybeam.ms
#
################################################################################

set -euo pipefail

echo "Starting Rapthor benchmark script on $(hostname) at $(date)"

# Load spack modules
# ---------------------------------------------------------------------------- #
module load py-rapthor@master py-ska-sdp-benchmark-monitor

# Configure
# ---------------------------------------------------------------------------- #
# Set environmental variables
# These are used for populating the parset template file.

# Full path to the input measurement set. 
export INPUT_MS_FULL_PATH=$(readlink -f "$INPUT_PATH/visibility.scan-400_applybeam.ms")

# Export TMPDIR - without this some steps in rapthor use the default which may
# run out of space - this is also substituted into parset file
export TMPDIR=/dev/shm

# Run Rapthor
# ---------------------------------------------------------------------------- #
# Create output directories
mkdir -p $WORK_PATH/logs

# Set paths to parset template file / output parset file
PARSET_TEMPLATE_PATH=$INPUT_PATH/rapthor_skao_benchmark_template.parset
PARSET_PATH=$INPUT_PATH/rapthor_skao_benchmark.parset

# Substitute the environment variables into the parset template
envsubst < $PARSET_TEMPLATE_PATH > $PARSET_PATH
# Show parset contents
echo "Generated parset file for run: $PARSET_PATH"
echo -e "Parset contents:\n\
-----------------------------------------------------------------------------\n\
$(cat $PARSET_PATH)\n\
-----------------------------------------------------------------------------\n"

# Start monitoring
benchmon-start --sys --sys-freq 1 --call --call-prof-freq 1 --save-dir $REPORT_PATH

# Run rapthor on parset
time rapthor $PARSET_PATH

# Stop monitoring
benchmon-stop

echo "Rapthor benchmark script completed at $(date)"
