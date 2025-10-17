#!/bin/bash
# SLURM batch script for rapthor benchmarking runs on the SKAO AWS cluster.
#
# SLURM settings:
#SBATCH --job-name=rapthor-benchmark
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time 196:00:00
#SBATCH --output=logs/%x-%j-leader-slurm.out
#SBATCH --error=logs/%x-%j-leader-slurm.out

# Specify spack environment version number
SPACK_TAG=2025.08.1

# NOTE: Working directory is set as WORK_PATH by run_bench.sh script:
# https://gitlab.com/ska-telescope/sdp/ska-sdp-spack/-/blob/sp-5859-add-bench-scripts/scripts/run_bench.sh

# Folder for benchmarking monitoring output is set as REPORT_PATH by
# run_bench.sh script

# Specify path to configuration file, which contains all other required paths
PARSET_PATH=$INPUT_PATH/rapthor.parset

# Export SLURM options for worker nodes for toil to use
# --cpus-per-task and --nodes are taken from the parset file
export TOIL_SLURM_ARGS="--exclusive --time 196:00:00" 
# Export partition list for SALLOC to use
export SALLOC_PARTITION=$PARTITION
# Export TMPDIR - without this some steps in rapthor use the default which may
# run out of space - set to same as tmp directories used in parset
export TMPDIR=$OUTPUT_PATH/tmp

# Create output directories
cd $INPUT_PATH
mkdir -p logs # Create directory for logs

# Enable use of modules from a tagged spack env
module use "/shared/fsx1/spack/modules/$SPACK_TAG/linux-ubuntu22.04-x86_64_v3"
echo "Using version $SPACK_TAG of spack environment"

# Load module for rapthor and its dependencies
module purge
module load py-rapthor
echo "Loaded $(rapthor --version)"

# Ensure bdsf can find libboost_numpy311.so.1.86.0
module load boost
module show boost
export LD_LIBRARY_PATH=$BOOST_ROOT/lib:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

# Load benchmarking tools
module load py-ska-sdp-benchmark-monitor

# Start monitoring
benchmon-multinode-start --save-dir $REPORT_PATH --system --sys --call

# Run rapthor on parset
rapthor $PARSET_PATH

# Stop monitoring
benchmon-multinode-stop

# Visualize monitoring results
benchmon-visu --recursive --cpu --cpu-all --mem --net --disk \
 --fig-fmt png --fig-dpi medium "$REPORT_PATH/benchmon_traces_$(hostname)"

