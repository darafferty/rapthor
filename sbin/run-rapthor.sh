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

# Always cleanup when the script exits
trap cleanup EXIT

# Color codes for terminal output. If the script is run in a terminal and
# the `NO_COLOR` environment variable is not set, the output will be colored.
if [[ -t 2 && ${TERM:-dumb} != dumb && -z ${NO_COLOR:-} ]]; then
  RED=$'\033[31m'
  YELLOW=$'\033[33m'
  GREEN=$'\033[32m'
  RESET=$'\033[0m'
else
  RED=''
  YELLOW=''
  GREEN=''
  RESET=''
fi

# Functions to print messages with different severity levels. The messages
# will be printed to standard error, and will be colored if the script is run
# in a terminal and the `NO_COLOR` environment variable is not set.
info() {
  printf '\n%sINFO: %s%s\n\n' "$GREEN" "$*" "$RESET" >&2
}

warning() {
  printf '\n%sWARNING: %s%s\n\n' "$YELLOW" "$*" "$RESET" >&2
}

error() {
  printf '\n%sERROR: %s%s\n\n' "$RED" "$*" "$RESET" >&2
}

die() {
  local status=$?
  error "$@"
  exit "$((status == 0 ? 1 : status))"
}

# Cleanup function to remove the staging directory when the script exits,
# regardless of whether it exits normally or due to an error.
# shellcheck disable=SC2317
cleanup() {
  local status=$?
  [ -n "${STAGING_DIR-}" ] && rm -rf "${STAGING_DIR}"
  [ "${status}" -eq 0 ] && info "run-rapthor.sh exited successfully"
  [ "${status}" -ne 0 ] && error "run-rapthor.sh exited with non-zero status: ${status}"
  return "${status}"
}

# Read the named option ($1) only from the parset's [global] section.
# Match option names case-insensitively, trim surrounding whitespace, and remove
# matching single or double quotes. Print only absolute paths; otherwise print nothing.
# Treat parset values as data; never evaluate them as shell expressions.
parset_path() {
  awk -v option="${1,,}" '
    /^[[:space:]]*\[/ {
      in_global = (tolower($0) ~ /^[[:space:]]*\[global\]/)
      next
    }
    in_global && tolower($0) ~ "^[[:space:]]*" option "[[:space:]]*[:=]" {
      sub(/^[^=:]*[:=][[:space:]]*/, "")
      sub(/[[:space:]]*$/, "")
      if (/^".*"$/ || /^\047.*\047$/)
        $0 = substr($0, 2, length($0) - 2)
      if (/^\//) {
        print
        exit
      }
    }
  ' "${PARSET}"
}

# Check if the parset file exists and determine the working directory from
# the parset file. Also load the rapthor environment.
setup() {
  info "**** Running on node $(hostname) ****"

  # Load the rapthor environment
  # shellcheck disable=SC1090
  . "${RAPTHOR_RC}" || die "Failed to source rapthor environment from ${RAPTHOR_RC}!"

  # Check if the parset file is provided as an argument
  [ $# -ne 1 ] && die "Usage: $0 <parset-file>"

  # check if parset exists and make it an absolute path
  local parset=${1}
  PARSET=$(readlink -e "${parset}") || die "Parset file ${parset} not found!"

  # Extract the working directory from the parset file, path must be absolute
  WORKING_DIR=$(parset_path dir_working)
  [ -n "${WORKING_DIR}" ] || die "Working directory invalid or not set in ${PARSET}!"

  # Create working directory if it does not exist, and check if it is writable.
  mkdir -p "${WORKING_DIR}" || die "Failed to create working directory ${WORKING_DIR}!"
  [ -w "${WORKING_DIR}" ] || die "Working directory ${WORKING_DIR} is not writable!"

  # Create a staging directory for the tar-ball artifacts.
  STAGING_DIR="$(mktemp -d tmp.XXXXXX -p "${WORKING_DIR}")" ||
    die "Failed to create staging directory in ${WORKING_DIR}!"

  # Print the software versions to the console for debugging purposes.
  info "**** C++ packages ****"
  cat "${RAPTHOR_INSTALL_DIR}/sw-versions.txt"
  info "**** Python packages ****"
  python -m pip list
}

# Run Rapthor with the specified parset file. If the execution fails, print
# a warning message, and update the RAPTHOR_STATUS variable.
run_rapthor() {
  info "Running Rapthor with parset file ${PARSET}"
  rapthor -v "${PARSET}" || {
    RAPTHOR_STATUS=$?
    warning "Rapthor execution failed, exit code: ${RAPTHOR_STATUS}"
  }
}

# Create a tar-ball of the logs, the parset file, and optionally the user-
# supplied skymodel file(s) and processing strategy. The tar-ball will
# be stored in the working directory, and its name will be of the form
# `rapthor_logs_<timestamp>.tar.gz`, where `<timestamp>` is the date and
# time when the tar-ball was created.
create_tarball() {
  info "Creating tar-ball of logs and configuration files"
  local tarball target
  tarball="${WORKING_DIR}/rapthor_logs_$(date +%Y%m%d-%H%M%S).tar.gz"
  ln -s "${PARSET}" "${STAGING_DIR}/parset"
  for name in input_skymodel apparent_skymodel strategy; do
    target=$(parset_path "${name}")
    [ -n "${target}" ] && ln -s "${target}" "${STAGING_DIR}/${name}"
  done
  ln -s "${WORKING_DIR}/logs" "${STAGING_DIR}/logs"

  if tar -C "${STAGING_DIR}" -chzf "${tarball}" .; then
    info "Tar-ball created successfully: ${tarball}"
  else
    warning "Errors occurred while creating ${tarball}: tar-ball may be incomplete"
  fi
}

setup "$@"
run_rapthor
create_tarball
exit ${RAPTHOR_STATUS}
