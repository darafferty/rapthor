#!/bin/bash

# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

# Script for testing the calibrate_init and calibrate update method
# in the idg proxies with test_calibrate.py.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export DATADIR=${DIR}/tmp/data
export MODELIMAGE="modelimage.fits"
export MSNAME="LOFAR_MOCK_3STATIONS.ms"

# Check whether DPPP (DP3) executable on $PATH
if ! [ -x "$(command -v DPPP)" ]; then
  echo 'Error: make sure the DPPP executable can be found on your path.'
  exit 1
fi

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

# Reduce size
if [ -d $DATADIR/$MSNAME ]
then
    echo "Already found a reduced size MS named ${MSNAME}"
else
    # Reduce the LOFAR_MOCK.ms even further by selecting just 3 baselines, write to ${MSNAME}
    DPPP msin=$DATADIR/LOFAR_MOCK.ms 'filter.baseline=[R]S30*&&[R]S30*' msout=$DATADIR/$MSNAME steps=[filter] filter.remove=true msout.overwrite=true
fi

# Download modelimage, if not yet done
if [ -f "$DATADIR/$MODELIMAGE" ]
then
    echo "${MODELIMAGE} already exists"
else
    ./../../scripts/download_modelimage.sh
fi

# Run the test
pytest -s --exitfirst ${DIR}/idgcaldpstep/test_calibrate.py