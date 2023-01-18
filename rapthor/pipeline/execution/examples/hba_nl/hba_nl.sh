#!/bin/sh -ex
. /project/rapthor/Software/rapthor/venv/bin/activate
cwltool \
  --copy-outputs \
  --leave-tmpdir \
  --outdir /project/rapthor/Share/rapthor/L667526 \
  --preserve-environment X509_USER_PROXY \
  --singularity \
  --tmpdir-prefix /project/rapthor/Share/rapthor/scratch/ \
  ../../hba_nl.cwl \
  hba_nl.json
