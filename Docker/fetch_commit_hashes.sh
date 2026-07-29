#!/bin/sh -e
git ls-remote https://gitlab.com/aroffringa/aoflagger.git     HEAD | awk '{ print "AOFLAGGER_COMMIT="$1 }'
git ls-remote https://github.com/casacore/casacore.git        HEAD | awk '{ print "CASACORE_COMMIT="$1 }'
git ls-remote https://git.astron.nl/RD/DP3.git                18e793a4 | awk '{ print "DP3_COMMIT="$1 }'
git ls-remote https://git.astron.nl/RD/EveryBeam.git          HEAD | awk '{ print "EVERYBEAM_COMMIT="$1 }'
git ls-remote https://git.astron.nl/RD/idg.git                HEAD | awk '{ print "IDG_COMMIT="$1 }'
git ls-remote https://github.com/casacore/python-casacore.git HEAD | awk '{ print "PYTHONCASACORE_COMMIT="$1 }'
# Keep libdirac compatible with Rapthor's GLib-free build.
printf '%s\n' 'SAGECAL_COMMIT=33d21c45000bf13e5e29077ba3413405c42c503f'
git ls-remote https://gitlab.com/aroffringa/wsclean.git       HEAD | awk '{ print "WSCLEAN_COMMIT="$1 }'
