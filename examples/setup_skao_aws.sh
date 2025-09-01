#!/bin/bash
################## Set up development environment for rapthor ##################
# This shell script will use spack to load rapthor dependencies not installed
# by pip and then create a virtual python env and install rapthor in editable
# mode so that code changes can be tested
 
########################## Specify key parameters #############################
 
# Specify spack environment version number
SPACK_TAG=2025.07.3
 
# Specify directory to rapthor repo
RAPTHOR_PATH=/path/to/rapthor/repo
 
###################### Do not edit below this line ############################
cd $RAPTHOR_PATH
 
# Enable use of modules from a tagged spack env
module use "/shared/fsx1/spack/modules/$SPACK_TAG/linux-ubuntu22.04-x86_64_v3"
echo "Using version $SPACK_TAG of spack environment"
 
# Load modules for rapthor's dependencies
module purge
module load python wsclean dp3 casacore py-casacore # everybeam idg aoflagger casacore py-casacore are loaded as dependencies of these
module load node-js # required for cwltool/rapthor
 
# Create and activate virtual env
python -m venv .venv
source .venv/bin/activate
 
# Upgrade pip and install rapthor in editable mode
pip install --upgrade pip
pip install --no-cache-dir --editable $RAPTHOR_PATH
 
# Install dev dependencies
pip install pytest mock pytest-cov tox