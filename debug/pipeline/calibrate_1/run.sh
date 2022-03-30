#!/bin/sh -ex
toil clean /project/rapthor/Share/rapthor.rap-423/HBA_short/pipelines/calibrate_1/jobstore
toil-cwl-runner \
  --singularity \
  --bypass-file-store \
  --batchSystem single_machine \
  --disableCaching \
  --defaultCores 4 \
  --defaultMemory 0 \
  --maxLocalJobs 1 \
  --jobStore /project/rapthor/Share/rapthor.rap-423/HBA_short/pipelines/calibrate_1/jobstore \
  --basedir /project/rapthor/Share/rapthor.rap-423/HBA_short/pipelines/calibrate_1 \
  --outdir /project/rapthor/Share/rapthor.rap-423/HBA_short/pipelines/calibrate_1 \
  --writeLogs /project/rapthor/Share/rapthor.rap-423/HBA_short/logs/calibrate_1 \
  --logLevel DEBUG \
  --maxLogFileSize 0 \
  --tmp-outdir-prefix /project/rapthor/Share/rapthor.rap-423/HBA_short/pipelines/calibrate_1/tmp/tmp. \
  --clean never \
  --cleanWorkDir never \
  --retryCount 0 \
  --servicePollingInterval 10 \
  --statePollingWait 10 \
  --stats \
  /project/rapthor/Share/rapthor.rap-423/HBA_short/pipelines/calibrate_1/pipeline_parset.cwl \
  /project/rapthor/Share/rapthor.rap-423/HBA_short/pipelines/calibrate_1/pipeline_inputs.yml
