#!/bin/bash

###############################################################################
# Rapthor Benchmark Script for SKAO AWS cluster
#
# This script is run by the `run_bench_all.sh` script in ska-sdp-spack/scripts/:
# https://gitlab.com/ska-telescope/sdp/ska-sdp-spack/-/blob/sp-5859-add-bench-scripts/scripts/run_bench.sh
# This loads the environmental variables set in the file
# `ska-sdp-spack/scripts/bench_specs/_ical.sh`, and executed the script by
# running the wrapper script `run_bench.sh`.

# Environment variables:
# ---------------------------------------------------------------------------- #
# The following are set in `run_bench.sh`:
#     INPUT_PATH   : Path containing input Measurement Set (*.ms),
#                    pipeline config file, and any extra inputs.
#     WORK_PATH    : Scratch/work directory
#     OUTPUT_PATH  : Directory where pipeline outputs will be written
#     REPORT_PATH  : Directory where reports will be copied
#     META_MODULE  : Spack meta-module to load dependencies
#
# The following are set by `_ical.sh`:
#     PARTITION    : SLURM partition to use
#     NODE_COUNT   : Number of nodes to use
#
# Parset and Strategy:
# ---------------------------------------------------------------------------- #
# The `run_bench.sh` wrapper script will synchronise the parset and strategy
# files into the INPUT_PATH before executing this script. The parset used for
# this run is available in the rapthor repository under
# `/examples/rapthor_skao_benchmark_template.parset`. 
# These files are also hosted on S3 at: 
#   s3://skao-sdp-testdata/gecko/ical-benchmark-input/extra-inputs/2025.10.24
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
# SLURM settings:
# ---------------------------------------------------------------------------- #
#SBATCH --job-name=sp-5859-rapthor-benchmark-test
#SBATCH --ntasks-per-node=1
#SBATCH --time 196:00:00
#SBATCH --output=%x-%j-leader-slurm.out
#SBATCH --error=%x-%j-leader-slurm.out
# ---------------------------------------------------------------------------- #
#
echo "Starting Rapthor benchmark script on $(hostname) at $(date)"

# Configure
# ---------------------------------------------------------------------------- #
# Set paths to parset template file / output parset file
PARSET_TEMPLATE_PATH=$INPUT_PATH/rapthor_skao_benchmark_template.parset
PARSET_PATH=$INPUT_PATH/rapthor_skao_benchmark.parset
# Export the full path to the input measurement set. This is needed for
# populating the environmental variables in the parset file template.
export INPUT_MS_FULL_PATH=$(readlink -f "$INPUT_PATH/visibility.scan-400.ms")
# Export TMPDIR - without this some steps in rapthor use the default which may
# run out of space - this is also substituted into parset file
export TMPDIR=$WORK_PATH/tmp

# Substitute the environment variables into the parset template
envsubst < $PARSET_TEMPLATE_PATH > $PARSET_PATH
echo "Generated parset file for run: $PARSET_PATH"

# Export SLURM options for worker nodes for toil to use
# --cpus-per-task and --nodes are taken from the parset file

export TOIL_SLURM_ARGS="--exclusive --time 196:00:00 -p $SLURM_JOB_PARTITION --nodes=$((SLURM_JOB_NUM_NODES -1))"
                    #    "-p $SLURM_JOB_PARTITION --nodes=$(SLURM_JOB_NUM_NODES -1)"
echo "TOIL_SLURM_ARGS: $TOIL_SLURM_ARGS"

# Export partition list for SALLOC to use
export SALLOC_PARTITION=$SLURM_JOB_PARTITION
# Set number of threads for numexpr
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_ON_NODE

# Change to working directory
cd $WORK_PATH

# Create output directories
mkdir -p logs

# Load spack modules
module load "$META_MODULE"

# Load module for rapthor and its dependencies
RAPTHOR_MODULES="py-rapthor boost py-ska-sdp-benchmark-monitor"
module load $RAPTHOR_MODULES
echo "Loaded $(rapthor --version)"

# Ensure bdsf can find libboost_numpy311.so.1.86.0
module show boost
export LD_LIBRARY_PATH=$BOOST_ROOT/lib:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

# Run Rapthor
# ---------------------------------------------------------------------------- #
# Start monitoring
benchmon-multinode-start --save-dir $REPORT_PATH --system --sys --call

# Run rapthor on parset
rapthor $PARSET_PATH

# Stop monitoring
benchmon-multinode-stop

# Visualize monitoring results
benchmon-visu --recursive --cpu --cpu-all --mem --net --disk \
 --fig-fmt png --fig-dpi medium "$REPORT_PATH/benchmon_traces_$(hostname)"

