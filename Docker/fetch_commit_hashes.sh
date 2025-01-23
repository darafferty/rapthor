#!/bin/sh -e
git ls-remote https://git.astron.nl/RD/idg.git            HEAD   | awk '{ print "IDG_COMMIT="$1 }'
git ls-remote https://git.astron.nl/RD/EveryBeam.git      v0.6.2 | awk '{ print "EVERYBEAM_COMMIT="$1 }'
git ls-remote https://github.com/aroffringa/dysco.git     HEAD   | awk '{ print "DYSCO_COMMIT="$1 }'
git ls-remote https://gitlab.com/aroffringa/aoflagger.git HEAD   | awk '{ print "AOFLAGGER_COMMIT="$1 }'
# git ls-remote https://gitlab.com/aroffringa/wsclean.git   v3.5   | awk '{ print "WSCLEAN_COMMIT="$1 }'
echo "WSCLEAN_COMMIT=ed6e7225532ce49222e04897d3d30184cd4a3bff"
git ls-remote https://github.com/nlesc-dirac/sagecal.git  HEAD   | awk '{ print "SAGECAL_COMMIT="$1 }'
# git ls-remote https://git.astron.nl/RD/DP3.git            HEAD   | awk '{ print "DP3_COMMIT="$1 }'
echo "DP3_COMMIT=86fba9228652932409845ebcf165dcc77f0dc4cb"
