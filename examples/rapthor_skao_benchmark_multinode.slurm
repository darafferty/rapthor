#!/bin/bash

###############################################################################
# Rapthor Benchmark Script for SKAO AWS cluster
#
# This script is intended to be run by the `run_bench_all.sh` script in
# ska-sdp-spack/scripts/:
# https://gitlab.com/ska-telescope/sdp/ska-sdp-spack/-/blob/sp-5859-add-bench-scripts/scripts/run_bench.sh
# The wrapper script loads the environmental variables from the file
# `ska-sdp-spack/scripts/bench_specs/_ical.sh`, and executes this script by
# running the benchmark wrapper script `run_bench.sh`.

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
# SLURM settings:
# ---------------------------------------------------------------------------- #
#SBATCH --job-name=sp-5859-rapthor-benchmark-test
#SBATCH --nodes=4                  # This must match max_nodes in the parset.
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=86         # Must be set. Should be less than the total
# number of cpus on the node to avoid resource contention
#SBATCH --time 168:00:00
#SBATCH --output=%x-%j-leader-slurm.out
#SBATCH --error=%x-%j-leader-slurm.out
# ---------------------------------------------------------------------------- #
#
echo "Starting Rapthor benchmark script on $(hostname) at $(date)"

# Load spack modules
# ---------------------------------------------------------------------------- #
echo "Loading meta-module: $META_MODULE"
module purge
module load "$META_MODULE"
# rapthor dependencies
RAPTHOR_MODULES="python git node-js \
    aoflagger boost casacore dp3 everybeam gcc-runtime glibc idg \
    py-astropy py-bdsf py-casacore py-jinja2 \
    py-losoto py-matplotlib py-mocpy py-numpy py-python-dateutil py-reproject \
    py-requests py-rtree py-scipy py-shapely py-toil python-venv wsclean \
    py-ska-sdp-benchmark-monitor"
module load $RAPTHOR_MODULES

# Ensure bdsf can find libboost_numpy311.so.1.86.0
module show boost
export LD_LIBRARY_PATH=$BOOST_ROOT/lib:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

# Install rapthor from branch
# ---------------------------------------------------------------------------- #
# Check repo status
git -C $CODE_PATH status --porcelain
echo "Installing rapthor from branch $(git -C $CODE_PATH rev-parse --abbrev-ref HEAD)"

# Change to working directory
cd $WORK_PATH

# Create and activate virtual environment
python -m venv $WORK_PATH/.venv
source $WORK_PATH/.venv/bin/activate

# Install using pip
INSTALL_LOG=$LOG_PATH/rapthor-install.log
pip install --upgrade pip > $INSTALL_LOG
pip install $CODE_PATH -v >> $INSTALL_LOG

# Check installation
echo "Installed $(rapthor --version)"
echo "Environment ready!"

# Configure
# ---------------------------------------------------------------------------- #
# Set environmental variables
# These are used for populating the parset template file.

# Full path to the input measurement set. 
export INPUT_MS_FULL_PATH=$(readlink -f "$INPUT_PATH/visibility.scan-400.ms")

# Export TMPDIR - without this some steps in rapthor use the default which may
# run out of space - this is also substituted into parset file
export TMPDIR=/dev/shm

# Options for toil worker nodes. These are substituted into the parset template.
export BATCH_SYSTEM="slurm_static"
export WORKER_NODE_COUNT=4

export SALLOC_PARTITION=$SLURM_SUBMIT_HOST
export TOIL_SLURM_ARGS="--exclusive --exact -p $SALLOC_PARTITION --time 24:00:00"
# Get the cpu count for the partition to set
# `max_nodes`, `cpus_per_task`, `max_cores`, `max_threads` in parset

# Set number of threads for numexpr
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK

# Print configuration
echo "Worker node configuration:"
echo "BATCH_SYSTEM: $BATCH_SYSTEM"
echo "SALLOC_PARTITION: $SALLOC_PARTITION"
echo "WORKER_NODE_COUNT: $WORKER_NODE_COUNT"
echo "SLURM_JOB_CPUS_PER_NODE: $SLURM_JOB_CPUS_PER_NODE"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "TOIL_SLURM_ARGS: $TOIL_SLURM_ARGS"

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

# Start monitoring on leader node
benchmon-multinode-start --save-dir $REPORT_PATH --system --sys --call

# Run rapthor on parset
time rapthor $PARSET_PATH

# Stop monitoring
benchmon-multinode-stop

# Visualize monitoring results
benchmon-visu --recursive --cpu --cpu-all --mem --net --disk --inline-call \
 --fig-fmt png --fig-dpi medium "$REPORT_PATH"

