#!/bin/bash

# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

ORIG_DIR=$(pwd)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set some environment variables
export DATADIR=${DIR}/tmp/data
export WORKDIR=${DIR}/tmp/workdir
export MSNAME="LOFAR_MOCK.ms"
export PATH="$PATH:${DIR}/common"
export LD_LIBRARY_PATH="${IDG_LIB_DIR}:${LD_LIBRARY_PATH}"
export PYTHONPATH="${IDG_PYTHONPATH}:${DIR}/common:$PYTHONPATH"
export COMMON=${DIR}/common

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
python3 -m pytest -s -v --exitfirst --junitxml=${ORIG_DIR}/test_griddegrid.xml ${DIR}/gridding/test_*.py
