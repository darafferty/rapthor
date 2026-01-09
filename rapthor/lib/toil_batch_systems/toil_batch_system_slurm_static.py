# Copyright (C) 2015-2021 Regents of the University of California
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file contains portions of code duplicated from SingleMachineBatchSystem
# The original source code can be found at:
# https://github.com/DataBiosphere/toil/blob/master/src/toil/batchSystems/singleMachine.py
# All copied code is subject to the apache 2.0 license which is
# included at the top of this file.
#
# This file also contains modifications that are not part of the original
# work.

# Implement SlurmStaticBatchSystem a batch system that launches jobs using srun
# and allows slurm to control all resource allocation via srun.
# This is a minal implementation that uses SingleMachineBatchSystem to do
# the bulk of the job management and simply adds a srun wrapper.

import logging
import os
import subprocess
import time
import traceback

from typing import Type, Optional
from toil.batchSystems.registry import add_batch_system_factory
from toil.batchSystems.singleMachine import SingleMachineBatchSystem, Info
from toil.batchSystems.abstractBatchSystem import AbstractBatchSystem

from toil.common import Config
from toil.job import (
    AcceleratorRequirement,
)

logger = logging.getLogger(__name__)


class SlurmStaticBatchSystem(SingleMachineBatchSystem):
    def __init__(
        self,
        config: Config,
        maxCores: float,
        maxMemory: float,
        maxDisk: int,
        max_jobs: Optional[int] = None,
    ) -> None:
        super().__init__(config, maxCores, maxMemory, maxDisk, max_jobs)
        # Avoid any altering of requested number of cores for jobs.
        # This will force that coreFractions in startChild == actual cores.
        self.minCores = 1
        self.scale = 1
        logger.info("Allocating SlurmStaticBatchSystem")

    def add_options(cls) -> None:
        """
        Add any options specific for this batch system.
        """
        # No options for now, just avoid the base class from adding its own.


    def _startChild(
        self,
        jobCommand,
        jobID,
        coreFractions,
        jobMemory,
        jobDisk,
        job_accelerators: list[AcceleratorRequirement],
        environment,
    ):
        """
        Start a child process for the given job.

        Just call all jobs with srun.
        Slurm/srun will do all resource management for us, no attempt
        is made by SlurmStaticBatchSystem to allocate or track resources.

        If the job is started, returns its PID.
        If the job fails to start, reports it as failed and returns False.
        """

        # Wrap the command we want to run with srun
        jobCommand = f"srun --ntasks=1 --cpus-per-task={coreFractions} {jobCommand}"
        popen = None
        startTime = time.time()
        acquired = []
        try:
            # Launch the job.
            # Use the same subprocess paramaters as SingleMachineBatchSystem
            # These are picked to make it easy for toil to manage/kill the
            # subprocesses when required.
            # See SingleMachineBatchSystem implementation for more details.
            logger.info("Attempting to run job command: %s", jobCommand)
            child_environment = dict(os.environ, **environment)
            popen = subprocess.Popen(
                jobCommand, shell=True, env=child_environment, start_new_session=True
            )
        except Exception:
            logger.error("Could not start job %s: %s", jobID, traceback.format_exc())

            # Report as failed.
            self.outputQueue.put(
                UpdatedBatchJobInfo(
                    jobID=jobID,
                    exitStatus=EXIT_STATUS_UNAVAILABLE_VALUE,
                    wallTime=0,
                    exitReason=None,
                )
            )

            # Complain it broke.
            return False
        else:
            # If the job did start, record it
            self.children[popen.pid] = popen
            # Make sure we can look it up by PID later
            self.childToJob[popen.pid] = jobID
            # Record that the job is running, and the resources it is using
            info = Info(startTime, popen, acquired, killIntended=False)
            self.runningJobs[jobID] = info
            logger.debug("Launched job %s as child %d", jobID, popen.pid)
            # Report success starting the job
            # Note that if a PID were somehow 0 it would look like False
            assert popen.pid != 0
            return popen.pid


# Register slurm_static batch system with toil
def slurm_static_batch_system_factory() -> Type[AbstractBatchSystem]:
    return SlurmStaticBatchSystem


add_batch_system_factory("slurm_static", slurm_static_batch_system_factory)
