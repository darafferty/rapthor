#!/bin/bash

# Runs the CI jobs

set -e

python3 -m pip install h5py pytest mock
python3 -m pip install .

cd test
pytest -v
