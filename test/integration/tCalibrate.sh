#!/bin/bash

# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

# Script for testing the calibrate_init and calibrate update method
# in the idg proxies with test_calibrate.py.

ORIG_DIR=$(pwd)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export DATADIR=${DIR}/tmp/data
export LD_LIBRARY_PATH="${IDG_LIB_DIR}:${LD_LIBRARY_PATH}"
export PYTHONPATH="${IDG_PYTHONPATH}:${DIR}/common:$PYTHONPATH"
export MODELIMAGE="modelimage.fits"
export MSNAME="LOFAR_MOCK_3STATIONS.ms"

# Check whether DP3 executable on $PATH
if ! [ -x "$(command -v DP3)" ]; then
  echo 'Error: make sure the DP3 executable can be found on your path.'
  exit 1
fi

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
    DP3 msin=$DATADIR/LOFAR_MOCK.ms 'filter.baseline=[R]S30*&&[R]S30*' msout=$DATADIR/$MSNAME steps=[filter] filter.remove=true msout.overwrite=true
fi

# Download modelimage, if not yet done
if [ -f "$DATADIR/$MODELIMAGE" ]
then
    echo "${MODELIMAGE} already exists"
else
    ./../../scripts/download_modelimage.sh
fi

# Run the test
pytest -s --exitfirst --junitxml=${ORIG_DIR}/test_calibrate.xml ${DIR}/idgcaldpstep/test_calibrate.py