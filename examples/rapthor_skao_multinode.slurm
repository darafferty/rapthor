#!/bin/bash
# Example SLURM batch script for running rapthor on multiple nodes on the SKAO AWS cluster.
# Adjust requested resources and job variables as necessary.
#
# SLURM settings:
#SBATCH --job-name==mn-rapthor
#SBATCH --partition=c7i-metal-24xl-ond   # Use spot partitions where possible.
#SBATCH --nodes=1                  # Use a single node by default.
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96          # For the leader node that starts jobs on other worker nodes
#SBATCH --time 24:00:00            # Set job timeout.
#SBATCH --output=logs/%x-%j-leader-slurm.out
#SBATCH --error=logs/%x-%j-leader-slurm.out

# Specify spack environment version number
SPACK_TAG=2025.07.3

# Specify working directory
WORK_DIR=/shared/fsx1/user.name

# Specify directory for benchmarking monitoring output
TRACES_DIR="$WORK_DIR/monitor/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_monitor"

# Specify path to configuration file, which contains all other required paths
PARSET_PATH=$WORK_DIR/rapthor.parset

# Export SLURM options for worker nodes for toil to use
# --cpus-per-task and --nodes are taken from the parset file
export TOIL_SLURM_ARGS="--partition=c7i-metal-24xl-ond --exclusive --time 24:00:00" # Use spot partitions where possible

# Export partition list for SALLOC to use
export SALLOC_PARTITION=c7i-metal-24xl-ond

# Export TMPDIR - without this some steps in rapthor use the default which may
# run out of space - set to same as tmp directories used in parset
export TMPDIR=$WORK_DIR/tmp


# -----------------------------------------------------------------------------
# Anything below should not be edited
# -----------------------------------------------------------------------------

# Create output directories
cd $WORK_DIR
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
benchmon_params="--save-dir $TRACES_DIR --system --sys"
benchmon-multinode-start $benchmon_params

# Run rapthor on parset
rapthor $PARSET_PATH

# Stop monitoring
benchmon-multinode-stop

# Visualize monitoring results
benchmon-visu --recursive --cpu --cpu-all --mem --net --disk --fig-fmt png --fig-dpi medium "$TRACES_DIR/benchmon_traces_$(hostname)"

