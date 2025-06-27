#!/bin/bash
# Example SLURM batch script for running rapthor on the SKAO AWS cluster.
# Adjust requested resources and job variables as necessary.
#
# SLURM settings:
#SBATCH --job-name=rapthor
#SBATCH --partition=c7i-24xl-spt   # Use spot partitions where possible.
#SBATCH --nodes=1                  # Use a single node by default.
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive                # Gain exclusive access to the nodes / avoid sharing.
#SBATCH -t 24:00:00                # Set job timeout.
#SBATCH --output=logs/%x-%j-slurm.out
#SBATCH --error=logs/%x-%j-slurm.out

# Specify spack environment version number
SPACK_TAG=2025.06.4

# Specify working directory
WORK_DIR=/shared/fsx1/user.name

# Specify path to configuration file, which contains all other required paths
PARSET_PATH=$WORK_DIR/parset_dir/rapthor.parset


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

# Run rapthor on parset
rapthor $PARSET_PATH
