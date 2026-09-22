#!/bin/sh -e
git ls-remote https://gitlab.com/aroffringa/aoflagger.git     HEAD | awk '{ print "AOFLAGGER_COMMIT="$1 }'
git ls-remote https://github.com/casacore/casacore.git        HEAD | awk '{ print "CASACORE_COMMIT="$1 }'
git ls-remote https://git.astron.nl/RD/DP3.git                HEAD | awk '{ print "DP3_COMMIT="$1 }'
# Temporary compatibility with DP3 master (< 0.9); update together with WSClean.
printf '%s\n' 'EVERYBEAM_COMMIT=882b7c0b1aefc76511815d7f5e1174f4861a6956' # v0.8.5
git ls-remote https://git.astron.nl/RD/idg.git                HEAD | awk '{ print "IDG_COMMIT="$1 }'
git ls-remote https://github.com/casacore/python-casacore.git HEAD | awk '{ print "PYTHONCASACORE_COMMIT="$1 }'
git ls-remote https://github.com/nlesc-dirac/sagecal.git      HEAD | awk '{ print "SAGECAL_COMMIT="$1 }'
# Last WSClean revision before the EveryBeam 0.9 API requirement.
printf '%s\n' 'WSCLEAN_COMMIT=2d5c1ed8d7559e5178d243661b408baa288681e6'
