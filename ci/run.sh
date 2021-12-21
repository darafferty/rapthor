#!/bin/bash

# Runs the CI job.
# For now there's one job, more jobs can be added later.

set -e

python3 setup.py install

# TODO RAP-327
# At the moment only one tests passes. After fixing the other two the single
# test should be removed and the commented out code should be enabled.
cd test
for f in *.py; do
	python3 $f
done

