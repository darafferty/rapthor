#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set some environment variables
export DATADIR=${DIR}/tmp/data
export WORKDIR=${DIR}/tmp/workdir
export MSNAME="LOFAR_MOCK.ms"
export PATH="$PATH:${DIR}/common"
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

${DIR}/gridding/test_gridding.py -v
