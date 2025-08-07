#!/bin/bash
# Example SLURM batch script for running rapthor on the SKAO AWS cluster.
# Adjust requested resources and job variables as necessary.
#
# SLURM settings:
#SBATCH --job-name=sn-rapthor
#SBATCH --partition=c7i-48xl-spt   # Use spot partitions where possible.
#SBATCH --nodes=1                  # Use a single node by default.
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive                # Gain exclusive access to the nodes / avoid sharing.
#SBATCH -t 24:00:00                # Set job timeout.
#SBATCH --output=logs/%x-%j-slurm.out
#SBATCH --error=logs/%x-%j-slurm.out

# Specify spack environment version number
SPACK_TAG=2025.07.3

# Specify working directory
WORK_DIR=/shared/fsx1/user.name

# Specify directory for benchmarking monitoring output
TRACES_DIR="$WORK_DIR/monitor/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_monitor"

# Specify path to configuration file, which contains all other required paths
PARSET_PATH=$WORK_DIR/parset_dir/rapthor.parset

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
benchmon_params="--save-dir $TRACES_DIR --system --sys --call"
benchmon-start $benchmon_params

# Run rapthor on parset
rapthor $PARSET_PATH

# Stop monitoring
benchmon-stop
 
# Visualize monitoring results
benchmon-visu --cpu --cpu-all --mem --net --disk --fig-fmt png --fig-dpi medium "$TRACES_DIR/benchmon_traces_$(hostname)"
