#!/bin/bash

# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set some environment variables
export DATADIR=${DIR}/tmp/data
export WORKDIR=${DIR}/tmp/workdir

export MODELIMAGE="modelimage.fits"
export MSNAME="LOFAR_MOCK_3STATIONS.ms"
export PATH="$PATH:${DIR}/common"
export COMMON=${DIR}/common

export MODELIMAGE_PATH=${DATADIR}/${MODELIMAGE}
export MS_PATH=${DATADIR}/${MSNAME}

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
    # Reduce the LOFAR_MOCK.ms even further by selecting just 3 baselines, write to ${MSNAME}
    DPPP msin=$DATADIR/LOFAR_MOCK.ms 'filter.baseline=[R]S30*&&[R]S30*' msout=$DATADIR/$MSNAME steps=[filter] filter.remove=true msout.overwrite=true
fi

# Download the modelimage.ms if it does not yet exist
if [ -f "$DATADIR/$MODELIMAGE" ]
then
    echo "${MODELIMAGE} already exists"
else
    ./../../scripts/download_modelimage.sh
fi

# Substitute variables in DPPP.parset
# Fill modulefile template and copy to AOFLAGGER_MODULEFILE_PATH
envsubst '${MODELIMAGE_PATH} ${MS_PATH}' < "${COMMON}/DPPP.parset.in" > "${WORKDIR}/DPPP.parset"

mkdir -p $WORKDIR
cd $WORKDIR

# Disable warnings due to HDF5 version mismatches
export HDF5_DISABLE_VERSION_CHECK=2

# Run the test
# pytest -s captures the print() statements
# TODO: check/add more fine grained log levels
pytest -s --exitfirst ${DIR}/idgcaldpstep/test_idgcalstep.py