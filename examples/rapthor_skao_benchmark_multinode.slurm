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
#SBATCH --nodes=1                  # Use a single node by default.
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time 48:00:00
#SBATCH --output=%x-%j-leader-slurm.out
#SBATCH --error=%x-%j-leader-slurm.out
# ---------------------------------------------------------------------------- #
#
echo "Starting Rapthor benchmark script on $(hostname) at $(date)"

# Configure
# ---------------------------------------------------------------------------- #
# Set environmental variables
# These are used for populating the parset template file.

# Full path to the input measurement set. 
export INPUT_MS_FULL_PATH=$(readlink -f "$INPUT_PATH/visibility.scan-400.ms")

# Export TMPDIR - without this some steps in rapthor use the default which may
# run out of space - this is also substituted into parset file
export TMPDIR=$WORK_PATH/tmp

# Options for toil worker nodes. These are substituted into the parset template.
# Partition list for toil SALLOC to use
export BATCH_SYSTEM="slurm"
export SALLOC_PARTITION=$SLURM_SUBMIT_HOST
export WORKER_NODE_COUNT=3
export TOIL_SLURM_ARGS="--time 48:00:00 -p $SALLOC_PARTITION -N $WORKER_NODE_COUNT --ntasks-per-node=5"
# Get the cpu count for the partition to set
#  `max_nodes`, `cpus_per_task`, `max_cores`, `max_threads` in parset
export CPUS_ON_NODE=$SLURM_JOB_CPUS_PER_NODE

# Set number of threads for numexpr
export NUMEXPR_MAX_THREADS=$CPUS_ON_NODE

# Print configuration
echo "Worker node configuration:"
echo "SALLOC_PARTITION: $SALLOC_PARTITION"
echo "WORKER_NODE_COUNT: $WORKER_NODE_COUNT"
echo "CPUS_ON_NODE: $CPUS_ON_NODE"
echo "TOIL_SLURM_ARGS: $TOIL_SLURM_ARGS"

# Set paths to parset template file / output parset file
PARSET_TEMPLATE_PATH=$INPUT_PATH/rapthor_skao_benchmark_template.parset
PARSET_PATH=$INPUT_PATH/rapthor_skao_benchmark.parset

# Substitute the environment variables into the parset template
envsubst < $PARSET_TEMPLATE_PATH > $PARSET_PATH
echo "Generated parset file for run: $PARSET_PATH"

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
# Start monitoring on leader node
BENCHMON_PARAMS="--save-dir $REPORT_PATH --system --sys --call"
benchmon-multinode-start $BENCHMON_PARAMS

# # Benchmon needs to be started on the worker nodes manually. We don't know 
# # which nodes will be allocated, so just monitor all nodes in the partition.
# ALL_NODES=($(sinfo -p $SALLOC_PARTITION -o "%n" -h))
# for NODE_NAME in ${ALL_NODES[@]}; do
#     srun --nodes=1 --ntasks=1 --nodelist="$NODE_NAME" --cpus-per-task=1 \
#         benchmon-start $BENCHMON_PARAMS 
# done

# Run rapthor on parset
time rapthor $PARSET_PATH

# Stop monitoring
benchmon-multinode-stop
# for NODE_NAME in ${ALL_NODES[@]}; do
#     srun --wait --nodes=1 --ntasks=1 --nodelist="$NODE_NAME" --cpus-per-task=1 \
#         benchmon-stop
# done

# Visualize monitoring results
benchmon-visu --recursive --cpu --cpu-all --mem --net --disk --inline-call \
 --fig-fmt png --fig-dpi medium "$REPORT_PATH"

