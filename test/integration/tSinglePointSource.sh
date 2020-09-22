#!/bin/bash
#
# Integration test for single point source, centered at the image center
#
# Make sure that: 
# - casacore/lib
# - aoflagger/lib
# - (boost/lib)
# - idg/lib
# are on your $LD_LIBRARY_PATH!

# Get full path to this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set some environment variables
export DATADIR=${DIR}/tmp/data
export WORKDIR=${DIR}/tmp/workdir
export MSNAME="LOFAR_MOCK.ms"
export PATH="$PATH:${DIR}/common"
export COMMON=${DIR}/common

function testrc {
    if [ $1 -eq 0 ]; then
        echo -e "\033[32mPassed\e[0m"
    else
        # Remove tmp directory and exit
        rm -rf ${DIR}/tmp
        echo -e "\033[31mFailed\e[0m"
        exit 1
    fi
}

function printtestname {
    echo ============================ >> log.txt
    echo -n $1 | tee -a log.txt
    echo -n "................"
    echo -e "\n============================" >> log.txt
}

function printpreparetestname {
    echo ============================ >> log.txt
    echo $1 >> log.txt
    echo -e "============================" >> log.txt
}

# Download measurement set into test/tmp/data directory (if needed)
cd $DIR
# ./../../scripts/download_lofar_ms.sh 
if [ -d $DATADIR/LOFAR_MOCK.ms ]
then
    echo "LOFAR_MOCK.ms already exists"
else
    ./../../scripts/download_lofar_ms.sh 
fi
mkdir -p $WORKDIR
cd $WORKDIR

# Prepare the testset
printpreparetestname prepare-testset-1
${DIR}/singlepointsource/preparetestset.py

printtestname "test degridding pointsource in center, stokes I"
${DIR}/singlepointsource/test_pointsource.py I
testrc $?

printtestname "test degridding pointsource in center, stokes Q"
${DIR}/singlepointsource/test_pointsource.py Q
testrc $?

printtestname "test degridding pointsource in center, stokes U"
${DIR}/singlepointsource/test_pointsource.py U
testrc $?

printtestname "test degridding pointsource in center, stokes V"
${DIR}/singlepointsource/test_pointsource.py V
testrc $?

# Remove tmp directory
# rm -rf ${DIR}/tmp