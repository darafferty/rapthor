#!/bin/bash -e
#
# shellcheck disable=SC1091
#
# Get version information from the Git repository, using the python package
# `setuptools_scm`.

# Determine top-level directory of the repository, and change directory to it.
ROOT=$(git -C "$(dirname "${0}")" rev-parse --show-toplevel)
cd "${ROOT}"

# Check if module `setuptools_scm` is avaible.
if ! python3 -c "import setuptools_scm" 2>/dev/null
then
  # Are we inside a virtual environment?
  if [ -z "${VIRTUAL_ENV}" ]
  then
    # Activate virtual environment if it exists in `venv` directory,
    # else create one first.
    if ! [ -r "venv/bin/activate" ]
    then
      python3 -m venv venv
    fi
    . venv/bin/activate
  fi
  # Check again if module `setuptools_scm` is avaible, now in the virtualenv
  # If not, install it
  if ! python3 -c "import setuptools_scm" 2>/dev/null
  then
    pip install --disable-pip-version-check --quiet --require-virtualenv setuptools_scm
  fi
fi

# Run the command to get the version information
python3 -c "from setuptools_scm import get_version; print(get_version())"
