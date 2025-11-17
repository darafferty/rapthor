#!/bin/sh -e
git ls-remote https://gitlab.com/aroffringa/aoflagger.git     HEAD | awk '{ print "AOFLAGGER_COMMIT="$1 }'
git ls-remote https://github.com/casacore/casacore.git        HEAD | awk '{ print "CASACORE_COMMIT="$1 }'
git ls-remote https://git.astron.nl/RD/DP3.git                HEAD | awk '{ print "DP3_COMMIT="$1 }'
git ls-remote https://git.astron.nl/RD/EveryBeam.git          HEAD | awk '{ print "EVERYBEAM_COMMIT="$1 }'
git ls-remote https://git.astron.nl/RD/idg.git                HEAD | awk '{ print "IDG_COMMIT="$1 }'
git ls-remote https://github.com/casacore/python-casacore.git HEAD | awk '{ print "PYTHONCASACORE_COMMIT="$1 }'
git ls-remote https://github.com/nlesc-dirac/sagecal.git      HEAD | awk '{ print "SAGECAL_COMMIT="$1 }'
git ls-remote https://gitlab.com/aroffringa/wsclean.git       HEAD | awk '{ print "WSCLEAN_COMMIT="$1 }'

# Temporary overrides
echo EVERYBEAM_COMMIT=3c8df96e760e3e357457ba65c30e012ddfc7a108
