#!/bin/bash

# Runs the CI jobs

set -e

python3 setup.py install

cd test
for f in *.py; do
	python3 $f
done

