#!/bin/sh -ex
#
# Convert an INI-file to a JSON-file.

if [ $# -lt 1 -o $# -gt 2 ]
then
  echo "Usage: ${0} <ini-file> [<json-file>]"
  exit 1
fi

INPUT="${1}"
OUTPUT="${2:-${INPUT}.json}"

# Check if input file exists. Bail out if not.
if ! [ -f "${INPUT}" ]
then
  echo "${INPUT}: No such file or directory"
  exit 1
fi

# Check if `jc`, a tool for converting INI to JSON, is available.
# If not, install it in a virtual environment.
if ! $(which jc > /dev/null)
then
  # Are we inside a virtual environment? If not, create one and activate it.
  if [ -z "${VIRTUAL_ENV}" ]
  then
    ROOT="$(git rev-parse --show-toplevel)"
    python3 -m venv "${ROOT}/venv"
    . "${ROOT}/venv/bin/activate"
  fi
  pip install --disable-pip-version-check --quiet --require-virtualenv jc
fi

# Do the actual conversion from INI to JSON
cat "${INPUT}" | jc --ini -mp > "${OUTPUT}"
