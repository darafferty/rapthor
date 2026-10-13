#!/usr/bin/env bash
#
#SBATCH --cpus-per-task=16
#SBATCH --constraint=amd
#
# Script to run Rapthor. The `SBATCH` directives specify the resources required
# for the job when submitted to a job scheduler (like SLURM). The script
# sets up the environment and runs Rapthor with the specified parset file.
# Afterwards, it will create a tar-ball of all the logs, the parset file,
# and optionally the user-supplied skymodel file(s) and processing strategy.
# Information about the location of these files will be extracted from the
# parset file. The tar-ball will be stored in the working directory that is
# specified in the parset file. The name of the tar-ball will be of the form
# `rapthor_logs_<timestamp>.tar.gz`, where `<timestamp>` is the date and
# time when the tar-ball was created.

set -euo pipefail

# Default location of the rapthor environment configuration file. This file
# will be sourced to set up the rapthor environment. The path can be overridden
# by setting the `RAPTHOR_RC` environment variable before running the script.
# For example:
#   `RAPTHOR_RC=/path/to/rapthor.rc run-rapthor.sh rapthor.parset`
RAPTHOR_RC=${RAPTHOR_RC:-/project/rapthor/Software/rapthor/etc/rapthor.rc}

# Variable to hold the exit status of the `rapthor` command.
RAPTHOR_STATUS=0

# Error function
error()
{
  echo -e "\nERROR: $@\n" >&2
  exit 1
}

# Warning function
warning()
{
  echo -e "\nWARNING: $@\n" >&2
}

# Check if the parset file exists and determine the working directory from
# the parset file. Also load the rapthor environment.
setup()
{
  echo -e "\n**** Running on node $(hostname) ****"

  # Check if the parset file is provided as an argument
  [ $# -ne 1 ] && error "Usage: $0 <parset-file>"

  # check if parset exists and make it an absolute path
  local parset=${1}
  PARSET=$(readlink -e "${parset}") || error "Parset file ${parset} not found!"

  # Extract the working directory from the parset file, path must be absolute
  WORKING_DIR=$(
    sed -n 's,^[[:space:]]*dir_working[[:space:]]*=[[:space:]]*/,/,p' "${PARSET}" \
      | sed 's,[[:space:]]*$,,'
  )
  [ -n "${WORKING_DIR}" ] || error "Working directory invalid or not set in ${PARSET}!"

  # Create working directory if it does not exist, and check if it is writable.
  mkdir -p "${WORKING_DIR}" || error "Failed to create working directory ${WORKING_DIR}!"
  [ -w "${WORKING_DIR}" ] || error "Working directory ${WORKING_DIR} is not writable!"

  # Load the rapthor environment
  . "${RAPTHOR_RC}"

  # Print the software versions to the console for debugging purposes.
  echo -e "\n**** C++ packages ****"
  cat "${RAPTHOR_INSTALL_DIR}/sw-versions.txt"
  echo -e "\n**** Python packages ****"
  python -m pip list
}

# Run Rapthor with the specified parset file. If the execution fails, print
# a warning message, and update the RAPTHOR_STATUS variable.
run_rapthor()
{
  echo -e "\nRunning Rapthor with parset file ${PARSET} ..."
  rapthor -v "${PARSET}" || {
    RAPTHOR_STATUS=$?;
    warning "Rapthor execution failed, exit code: ${RAPTHOR_STATUS}";
  }
}

# Create a link to the given target with the given link name. Existing links
# will be overwritten. If the link name is not provided, the link will be
# created in the current working directory with the same name as the target.
# Return with an error status if the target does not exist or the link could
# not be created.
make_symlink()
{
  local target=${1}
  local link_name=${2}
  [ -n "${target}" ] || return 1
  [ -n "${link_name}" ] || link_name=$(basename "${target}")
  target=$(readlink -e "${target}") || {
    warning "Target file ${target} does not exist"
    return 1
  }
  ln -sfn "${target}" "${link_name}" || {
    warning "Failed to create symbolic link ${link_name} to target ${target}"
    return 1
  }
}

# Create a tar-ball of the logs, the parset file, and optionally the user-
# supplied skymodel file(s) and processing strategy. The tar-ball will
# be stored in the working directory, and its name will be of the form
# `rapthor_logs_<timestamp>.tar.gz`, where `<timestamp>` is the date and
# time when the tar-ball was created.
create_tarball()
{
  echo -e "\nCollecting logs ..."
  local timestamp=$(date +%Y%m%d_%H%M%S)
  local tarball_name="rapthor_logs_${timestamp}.tar.gz"
  local tarball_path="${WORKING_DIR}/${tarball_name}"
  local tarball_contents=()
  make_symlink "${PARSET}" "${WORKING_DIR}/parset" && tarball_contents+=("parset")
  for name in input_skymodel apparent_skymodel strategy; do
    local file=$(
      sed -n "s,^[[:space:]]*${name}[[:space:]]*=[[:space:]]*/,,p" "${PARSET}"
    )
    make_symlink "${file}" "${WORKING_DIR}/${name}" && tarball_contents+=("${name}")
  done
  [ -d "${WORKING_DIR}/logs" ] && tarball_contents+=("logs")
  echo -e "\nCreating tar-ball of logs and configuration files ..."
  tar -C "${WORKING_DIR}" -chzf "${tarball_path}" "${tarball_contents[@]}"
  echo -e "\nTar-ball created successfully: ${tarball_path}"
}

setup "$@"
run_rapthor
create_tarball
exit ${RAPTHOR_STATUS}
