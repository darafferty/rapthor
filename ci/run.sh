#!/bin/bash

# Runs the CI jobs

set -e

python3 -m pip install h5py pytest
python3 -m pip install .

cd test
pytest -v
