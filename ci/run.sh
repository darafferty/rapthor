#!/bin/bash

# Runs the CI jobs

set -e

python3 -m pip install .

cd test
for f in *.py; do
	python3 $f
done

