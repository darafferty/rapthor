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
#SBATCH --job-name=sp-5859-rapthor-long-path-test
#SBATCH --partition=any-7i-metal-24xl-spt
#SBATCH --exclusive
#SBATCH --nodes=1                  # Use a single node by default.
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=86
#SBATCH --time 168:00:00
#SBATCH --output=%x-%j-leader-slurm.out
#SBATCH --error=%x-%j-leader-slurm.out
# ---------------------------------------------------------------------------- #
#
echo "Starting Rapthor benchmark script on $(hostname) at $(date)"

# Configure
# ---------------------------------------------------------------------------- #
# Set environmental variables
# These are used for populating the parset template file.

export INPUT_PATH=/shared/fsx1/hannes/gec-182
WORK_PATH=$INPUT_PATH/test-path-fix-with-really-really-long-name-just-to-make-extra-sure-this-works

if [ -e $WORK_PATH ]; then
    for ((i=1;;i++)); do
        if [ ! -e $WORK_PATH-$i ]; then break; fi
    done
    WORK_PATH=$WORK_PATH-$i
else
    mkdir -p $WORK_PATH
fi

export WORK_PATH=$WORK_PATH
export CODE_PATH=/home/hannes.breytenbach/repos/rapthor
# REPORT_PATH=$WORK_PATH/report

# Full path to the input measurement set. 
export INPUT_MS_FULL_PATH="/shared/fsx1/shared/ical_workshop/data/visibility.scan-400_applybeam.ms"

# Export TMPDIR - without this some steps in rapthor use the default which may
# run out of space - this is also substituted into parset file
export TMPDIR=$WORK_PATH/tmp

# Get the cpu count for the partition to set
#  `max_nodes`, `cpus_per_task`, `max_cores`, `max_threads` in parset
export BATCH_SYSTEM="single_machine"
export WORKER_NODE_COUNT=1
export CPUS_ON_NODE=46

# Set number of threads for numexpr
export NUMEXPR_MAX_THREADS=46

# Set paths to parset template file / output parset file
PARSET_TEMPLATE_PATH=$INPUT_PATH/rapthor_ws_template.parset
# PARSET_TEMPLATE_PATH=/shared/fsx1/shared/ical-bench/rapthor_skao_benchmark_template.parset
PARSET_PATH=$INPUT_PATH/rapthor_ws_test.parset

# Substitute the environment variables into the parset template
envsubst < $PARSET_TEMPLATE_PATH > $PARSET_PATH
echo "Generated parset file for run: $PARSET_PATH"

# Create output directories
mkdir -p "$WORK_PATH/logs" $REPORT_PATH

# Load spack modules
module purge
export MODULEPATH=/shared/fsx1/shared/metamodules
META_MODULE=ska-sdp-spack/2025.10.3
module load "$META_MODULE"

# Load module for rapthor and its dependencies
RAPTHOR_MODULES="python node-js \
    aoflagger boost casacore dp3 everybeam gcc-runtime glibc idg \
    py-astropy py-bdsf py-casacore py-jinja2 \
    py-losoto py-matplotlib py-mocpy py-numpy py-python-dateutil py-reproject \
    py-requests py-rtree py-scipy py-shapely py-toil python python-venv wsclean"
module load $RAPTHOR_MODULES

# Change to working directory
cd $CODE_PATH
git checkout gec-182-test-benchmark

if [ -f $CODE_PATH/.venv/bin/activate ]; then
  # Activate existing virtual environment
  echo "Activating existing virtual environment." >&2
  source $CODE_PATH/.venv/bin/activate
else
  # Create virtual environment
  echo "WARNING: virtual environment not found. Creating a new one..." >&2
  python -m venv $CODE_PATH/.venv
  source $CODE_PATH/.venv/bin/activate

  # Install poetry using pip
  cd $CODE_PATH
  pip install --upgrade pip
  pip install -e . -v

  # Check installation
  echo "RAPTHOR VERSION $(rapthor --version)"
  echo "Environment ready!"
fi

# Ensure bdsf can find libboost_numpy311.so.1.86.0
module show boost
export LD_LIBRARY_PATH=$BOOST_ROOT/lib:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

# Run Rapthor
# ---------------------------------------------------------------------------- #
# Start monitoring
# benchmon-start --save-dir $REPORT_PATH --system --sys --call

# Change to working directory
cd $WORK_PATH

# Run rapthor on parset
time rapthor $PARSET_PATH

# Stop monitoring
# benchmon-stop

# Visualize monitoring results
# benchmon-visu --recursive --cpu --cpu-all --mem --net --disk --inline-call \
#     --fig-fmt png --fig-dpi medium "$REPORT_PATH"

