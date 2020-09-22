#!/bin/bash
#
# Author: Jakob Maljaars
# Email: jakob.maljaars_@_stcorp.nl

# Script for downloading a mock LOFAR Measurement set

set -e

SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_PATH

# Move up to parent folder containing the source
cd ..
mkdir -p test/integration/tmp/data
cd test/integration/tmp/data

LOFAR_MOCK_ARCHIVE=LOFAR_ARCHIVE.tar.bz2
LOFAR_MOCK_MS=LOFAR_MOCK.ms

if [ ! -f "$LOFAR_MOCK_ARCHIVE" ]; then
    wget -q https://www.astron.nl/citt/EveryBeam/L258627-150-timesteps.tar.bz2 -O $LOFAR_MOCK_ARCHIVE
fi

if [ -d $LOFAR_MOCK_MS ]
then
    echo "Directory already exists"
else
    mkdir $LOFAR_MOCK_MS
fi

tar -xf $LOFAR_MOCK_ARCHIVE  -C $LOFAR_MOCK_MS --strip-components=1
rm $LOFAR_MOCK_ARCHIVE