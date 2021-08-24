#!/bin/bash

# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set some environment variables
export DATADIR=${DIR}/tmp/data
export WORKDIR=${DIR}/tmp/workdir
export MSNAME="LOFAR_MOCK.ms"
export PATH="$PATH:${DIR}/common"
export PYTHONPATH="${DIR}/common:$PYTHONPATH"
export COMMON=${DIR}/common

# Get the python path to the idg pybindings
if [ -z "${IDG_LIB}" ] ;
then
      echo "\$IDG_LIB not in environment variables. Please do so!"
      exit 1;
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
IDG_PYTHON_PATH=${IDG_LIB}/python${PYTHON_VERSION}/site-packages
export PYTHONPATH=${IDG_PYTHON_PATH}:${PYTHONPATH}

# Download measurement set into test/tmp/data directory (if needed)
cd $DIR
if [ -d $DATADIR/LOFAR_MOCK.ms ]
then
    echo "LOFAR_MOCK.ms already exists"
else
    ./../../scripts/download_lofar_ms.sh
fi
mkdir -p $WORKDIR
cd $WORKDIR

# NOTE: it is required that test_gridding.py runs
# before test_pygridding.py, since test_pygridding assumes
# certain (fits) files to be present.
# TODO: this needs to be setup a bit more generic
PYTEST=$(which pytest-3 || echo "pytest")
${PYTEST} -s -v --exitfirst ${DIR}/gridding/test_*.py
