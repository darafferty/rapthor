#!/usr/bin/env bash
# Script to set up the compute environment and install the pipeline
set -eo pipefail

echo "Setting up pipeline environment..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get repository directory
if [ -n "${1-}" ]; then
    REPO_DIR="$(cd "$1" && pwd)"
else
    REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
echo "Repository directory is $REPO_DIR"

# check that REPO_DIR exists
if [ ! -d "$REPO_DIR" ]; then
    echo "Repository directory $REPO_DIR does not exist!" >&2
    exit 1
fi
# check that REPO_DIR is a git repository
echo $(git rev-parse --show-toplevel -C $REPO_DIR) >/dev/null 2>&1 || {
    echo "Repository directory $REPO_DIR is not a git repository!" >&2
    exit 1
}

# ------------------- Load modules and set up environment ----------------------

# Use latest spack if $SPACK_TAG not set in the environment
export MODULEPATH=/shared/fsx1/shared/metamodules

# Check if SPACK_TAG is set
if [ -z "$SPACK_TAG" ]; then
    echo "SPACK_TAG is unset"
    export SPACK_TAG=$(ls -1 $MODULEPATH/ska-sdp-spack/ | tail -n 1)
    echo "Using the latest ska-sdp-spack version: $SPACK_TAG"
else
    echo "Using ska-sdp-spack version: $SPACK_TAG"
fi
# Define meta module from tag
META_MODULE=ska-sdp-spack/$SPACK_TAG

# Load spack modules
module purge
module load "$META_MODULE"
echo "Loaded meta-module $META_MODULE"

# Load module for wsclean and dp3
module load wsclean dp3 python
echo "Loaded $(wsclean --version)"
echo "Loaded $(DP3 --version)"

# Create and activate virtual env
if [ -f $REPO_DIR/.venv/bin/activate ]; then
    # Activate existing virtual environment
    echo "Activating existing virtual environment." >&2
    source $REPO_DIR/.venv/bin/activate
else
    # Create virtual environment
    echo "WARNING: virtual environment not found. Creating a new one..." >&2
    python -m venv $REPO_DIR/.venv
    source $REPO_DIR/.venv/bin/activate

    # Install Python dependencies using pip
    pip install --upgrade pip
    pip install --no-binary=bdsf bdsf
    pip install -e $REPO_DIR

    # Check installation
    python -c "import rapthor"
    echo "Environment ready!"
fi
