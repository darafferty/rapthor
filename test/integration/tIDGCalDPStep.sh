#!/bin/bash

# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set some environment variables
export DATADIR=${DIR}/tmp/data
export WORKDIR=${DIR}/tmp/workdir

export MODELIMAGE="modelimage.fits"
export MSNAME="LOFAR_MOCK_15STATIONS.ms"
export PATH="$PATH:${DIR}/common"
export COMMON=${DIR}/common

export MODELIMAGE_PATH=${DATADIR}/${MODELIMAGE}
export MS_PATH=${DATADIR}/${MSNAME}

# Get the python path to DP3 pybindings
if [ -z "${DP3_LIB}" ] ;
then
      echo "\$DP3_LIB not in environment variables. Please do so!"
      exit 1;
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
DP3_PYTHON_PATH=${DP3_LIB}/python${PYTHON_VERSION}/site-packages

# Get the python path to the idg pybindings
if [ -z "${IDG_LIB}" ] ;
then
      echo "\$IDG_LIB not in environment variables. Please do so!"
      exit 1;
fi
IDG_PYTHON_PATH=${IDG_LIB}/python${PYTHON_VERSION}/site-packages

# Set python path
export PYTHONPATH=${IDG_PYTHON_PATH}:${DP3_PYTHON_PATH}:${PYTHONPATH}

# Download measurement set into test/tmp/data directory (if needed)
cd $DIR
if [ -d $DATADIR/LOFAR_MOCK.ms ]
then
    echo "LOFAR_MOCK.ms already exists"
else
    ./../../scripts/download_lofar_ms.sh
fi

if [ -d $DATADIR/$MSNAME ]
then
    echo "Already found a reduced size MS named ${MSNAME}"
else
    # Reduce the LOFAR_MOCK.ms even further by selecting 15 statios, write to ${MSNAME}
    DP3 msin=$DATADIR/LOFAR_MOCK.ms 'filter.baseline=0,10,20,30,40,50,52~60&' msout=$DATADIR/$MSNAME steps=[filter] filter.remove=true msout.overwrite=true
fi

# Download the modelimage.ms if it does not yet exist
if [ -f "$DATADIR/$MODELIMAGE" ]
then
    echo "${MODELIMAGE} already exists"
else
    ./../../scripts/download_modelimage.sh
fi

mkdir -p $WORKDIR
cd $WORKDIR

# Disable warnings due to HDF5 version mismatches
export HDF5_DISABLE_VERSION_CHECK=2

echo ${PYTHONPATH}
echo ${DP3_PYTHON_PATH}
ls ${DP3_PYTHON_PATH}

# Run the test
# pytest -s captures the print() statements
pytest -s --exitfirst ${DIR}/idgcaldpstep/test_idgcalstep.py