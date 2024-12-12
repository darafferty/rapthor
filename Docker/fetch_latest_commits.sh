#!/bin/sh -e
git ls-remote https://git.astron.nl/RD/idg.git HEAD            | awk '{ print "IDG_COMMIT="$1 }'
git ls-remote https://git.astron.nl/RD/EveryBeam.git v0.6.2    | awk '{ print "EVERYBEAM_COMMIT="$1 }'
git ls-remote https://github.com/aroffringa/dysco.git HEAD     | awk '{ print "DYSCO_COMMIT="$1 }'
git ls-remote https://gitlab.com/aroffringa/aoflagger.git HEAD | awk '{ print "AOFLAGGER_COMMIT="$1 }'
git ls-remote https://gitlab.com/aroffringa/wsclean.git HEAD   | awk '{ print "WSCLEAN_COMMIT="$1 }'
git ls-remote https://github.com/nlesc-dirac/sagecal.git HEAD  | awk '{ print "SAGECAL_COMMIT="$1 }'
git ls-remote https://git.astron.nl/RD/DP3.git HEAD            | awk '{ print "DP3_COMMIT="$1 }'
