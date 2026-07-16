#!/bin/sh -e
git ls-remote https://gitlab.com/aroffringa/aoflagger.git     HEAD | awk '{ print "AOFLAGGER_COMMIT="$1 }'
git ls-remote https://github.com/casacore/casacore.git        HEAD | awk '{ print "CASACORE_COMMIT="$1 }'
printf '%s\n' 'DP3_COMMIT=5cc94dd01f091ffa7ebeef92fbfd942a86c73021'
git ls-remote https://git.astron.nl/RD/EveryBeam.git          HEAD | awk '{ print "EVERYBEAM_COMMIT="$1 }'
git ls-remote https://git.astron.nl/RD/idg.git                HEAD | awk '{ print "IDG_COMMIT="$1 }'
git ls-remote https://github.com/casacore/python-casacore.git HEAD | awk '{ print "PYTHONCASACORE_COMMIT="$1 }'
# Keep libdirac compatible with Rapthor's GLib-free build.
printf '%s\n' 'SAGECAL_COMMIT=33d21c45000bf13e5e29077ba3413405c42c503f'
git ls-remote https://gitlab.com/aroffringa/wsclean.git       HEAD | awk '{ print "WSCLEAN_COMMIT="$1 }'
printf '%s\n' 'WSCLEAN_COMMIT=8bc500d20ee6fdb400ecdde7d8e6d3898836ba2c'
