#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --constraint=amd
#
# Script to run Rapthor. The `SBATCH` directives specify the resources required
# for the job when submitted to a job scheduler (like SLURM). The script sets
# up the environment and runs Rapthor with the specified parset file. 
# Afterwards, it will create a tar-ball of all the logs, the parset file, and 
# optionally the user-supplied skymodel file(s) and processing strategy.
# Information about the location of these files will be extracted from the 
# parset file. The tar-ball will be stored in the working directory that is
# specified in the parset file. The name of the tar-ball will be of the form
# `rapthor_logs_<timestamp>.tar.gz`, where `<timestamp>` is the date and time
# when the tar-ball was created.

# Error function
error()
{
  echo -e "ERROR: $@" >&2
  exit 1
}

# Check if the parset file exists and determine the working directory from the 
# parset file. Also load the rapthor environment.
setup()
{
  echo -e "\n**** Running on node $(hostname) ****"

  # check if parset exists and make it an absolute path
  local parset=${1:-rapthor.parset}
  PARSET=$(readlink -e ${parset}) || error "Parset file ${parset} not found!"

  # Extract the working directory from the parset file.
  WORKING_DIR=$(
    sed -n 's,^[[:space:]]*dir_working[[:space:]]*=[[:space:]]*,,p' ${PARSET}
  )
  [ -n "${WORKING_DIR}" ] || error "Working directory not set in ${PARSET}!"
  [ -d "${WORKING_DIR}" ] || error "Working directory ${WORKING_DIR} does not exist!"

  # Load the rapthor environment
  . /project/rapthor/Software/rapthor/etc/rapthor.rc
}

run_rapthor()
{
  # Print the software versions to the console for logging purposes.
  echo -e "\n**** Software versions ****"
  cat ${RAPTHOR_INSTALL_DIR}/sw-versions.txt
  echo -e "\nRunning Rapthor with parset file ${PARSET} ..."
  rapthor -v ${PARSET}
}

create_tarball()
{
  local timestamp=$(date +%Y%m%d_%H%M%S)
  local tarball_name="rapthor_logs_${timestamp}.tar.gz"
  local tarball_path="${WORKING_DIR}/${tarball_name}"
  local tarball_contents="logs"
  for name in input_skymodel apparent_skymodel strategy; do
    file=$(
      sed -n "s/^[[:space:]]*${name}[[:space:]]*=[[:space:]]*//p" ${PARSET} | \
      sed -n "/\//p"
    )
    # Create a symbolic link to each file in the working directory, so that
    # they are included in the tar-ball without including the full path.
    ln -sf ${file} ${WORKING_DIR} 2>/dev/null || true
    tarball_contents="${tarball_contents} $(basename ${file})"
  done
  echo -e "\nCreating tar-ball of logs and configuration files ..."
  tar -C ${WORKING_DIR} -chzf ${tarball_path} ${tarball_contents}
  echo -e "\nTar-ball created successfully: ${tarball_path}"
}

setup $@
run_rapthor
create_tarball
