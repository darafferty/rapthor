"""
Classes that wrap the CWL runners that Rapthor supports.
"""
from __future__ import annotations
import logging
import os
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rapthor.lib.operation import Operation

logger = logging.getLogger("rapthor:cwlrunner")

class CWLRunner:
    """
    Base class.
    CWLRunner should be used inside a context. This will ensure that the object is
    properly configured when its `run()` method is invoked. We need to have access
    to some of the settings in the `operation` that is calling us.
    """
    def __init__(self, operation: Operation) -> None:
        """
        Initalizer
        """
        logger.debug("CWLRunner.__init__()")
        self.args = []
        self.command = None
        self.operation = operation

    def __enter__(self) -> "CWLRunner":
        """
        Called when entering a context.
        """
        logger.debug("CWLRunner.__enter__()")
        self.setup()
        return self

    def __exit__(self, *_) -> None:
        """
        Called when exiting a context.
        """
        logger.debug("CWLRunner.__exit__()")
        self.teardown()

    def _create_mpi_config_file(self) -> None:
        """
        Create the config file for MPI jobs and add the required args
        """
        if self.operation.batch_system == 'slurm':
            mpi_config_lines = [
                "runner: 'mpirun'",
                "nproc_flag: '-np'",
                "extra_flags: ['--map-by node']"
            ]
        else:
            mpi_config_lines = [
                "runner: 'mpi_runner.sh'",
                "nproc_flag: '-N'",
                "extra_flags: ['mpirun', '--map-by node']"
            ]
            logger.warning('MPI support for non-Slurm clusters is experimental. '
                           'Please report any issues encountered.')
        with open(self.operation.mpi_config_file, 'w') as cfg_file:
            cfg_file.write('\n'.join(mpi_config_lines))

    def _delete_mpi_config_file(self) -> None:
        """
        Delete the MPI config file
        """
        os.remove(self.operation.mpi_config_file)

    def setup(self) -> None:
        """
        Prepare runner for running. Set up the list of arguments to pass to the
        actual CWL runner.
        Derived classes can add their own arguments by overriding this method.
        """
        logger.debug("CWLRunner.setup()")
        self.args.extend(["--outdir", self.operation.pipeline_working_dir])
        if self.operation.container is not None:
            # If the container is Docker, no extra args are needed. For other
            # containers, set the required args
            if self.operation.container == 'singularity':
                self.args.extend(['--singularity'])
            elif self.operation.container == 'udocker':
                self.args.extend(['--user-space-docker-cmd', 'udocker'])
        else:
            self.args.extend(['--no-container'])
            self.args.extend(['--preserve-entire-environment'])
        if self.operation.scratch_dir is not None:
            prefix = os.path.join(self.operation.scratch_dir, self.command + '.')
            self.args.extend(['--tmpdir-prefix', prefix])
            self.args.extend(['--tmp-outdir-prefix', prefix])
        if self.operation.field.use_mpi:
            self._create_mpi_config_file()
            self.args.extend(['--mpi-config-file', self.operation.mpi_config_file])
            self.args.extend(['--enable-ext'])

    def teardown(self) -> None:
        """
        Clean up after the runner has run.
        """
        logger.debug("CWLRunner.teardown()")
        if self.operation.field.use_mpi:
            self._delete_mpi_config_file()

    def run(self) -> bool:
        """
        Start the runner in a subprocess.
        Every CWL runner requires two input files:
          - the CWL workflow, provided by `self.operation.pipeline_parset_file`
          - inputs for the CWL workflow, provided by `self.operation.pipeline_inputs_file`
        Every CWL runner is supposed to print to:
          - `stdout`: a JSON file of the generated outputs
          - `stderr`: workflow diagnostics
        These streams are redirected:
          - `stdout` -> `self.operation.pipeline_outputs_file`
          - `stderr` -> `self.operation.pipeline_log_file`
        """
        print("**** self.operation.debug_workflow:", self.operation.debug_workflow)
        logger.debug("CWLRunner.run()")
        if self.command is None:
            raise RuntimeError(
                "Don't know how to start CWL runner {}".format(self.__class__.__name__)
            )
        args = [self.command] + self.args
        # for item in self.args.items():
        #     args.extend(item)
        args.extend([self.operation.pipeline_parset_file,
                     self.operation.pipeline_inputs_file])
        logger.debug("Executing command: %s", args)
        with open(self.operation.pipeline_outputs_file, 'w') as stdout, \
             open(self.operation.pipeline_log_file, 'w') as stderr:
            try:
                result = subprocess.run(args=args, stdout=stdout, stderr=stderr, check=True)
                logger.debug(str(result))
                return True
            except subprocess.CalledProcessError as err:
                logger.critical(str(err))
                return False


class ToilRunner(CWLRunner):
    """
    Wrapper class for the Toil CWL runner
    """
    def __init__(self, operation: Operation) -> None:
        """
        Initializer
        """
        logger.debug("ToilRunner.__init__()")
        super().__init__(operation)
        self.command = "toil-cwl-runner"
        self.toil_env_variables = {
            "TOIL_SLURM_ARGS": "--export=ALL"
        }

    def setup(self):
        """
        Prepare runner for running. Adds some additional preprations to base class.
        """
        logger.debug("ToilRunner.setup()")
        super().setup()
        self.args.extend(['--batchSystem', self.operation.batch_system])
        if self.operation.batch_system == 'slurm':
            self.args.extend(['--disableCaching'])
            self.args.extend(['--defaultCores', str(self.operation.cpus_per_task)])
            self.args.extend(['--defaultMemory', self.operation.mem_per_node_gb])
        self.args.extend(['--maxLocalJobs', str(self.operation.max_nodes)])
        self.args.extend(['--jobStore', self.operation.jobstore])
        if os.path.exists(self.operation.jobstore):
            self.args.extend(['--restart'])
        self.args.extend(['--writeLogs', self.operation.log_dir])
        self.args.extend(['--writeLogsFromAllJobs'])  # also keep logs of successful jobs
        self.args.extend(['--maxLogFileSize', '0'])  # disable truncation of log files
        if self.operation.scratch_dir is not None:
            # Note: add a trailing directory separator, required by Toil v5.3+
            self.args.extend(['--workDir', os.path.join(self.operation.scratch_dir, '')])
        self.args.extend(['--clean', 'never'])  # preserves the job store for future runs
        self.args.extend(['--servicePollingInterval', '10'])
        self.args.extend(['--stats'])
        if self.operation.debug_workflow:
            self.args.extend(['--cleanWorkDir', 'never'])
            self.args.extend(['--debugWorker'])  # NOTE: stdout/stderr are not redirected to the log
            self.args.extend(['--logDebug'])
        os.environ.update(self.toil_env_variables)

    def teardown(self):
        """
        Clean up after the runner has run.
        """
        logger.debug("ToilRunner.teardown()")
        for key in self.toil_env_variables:
            del os.environ[key]
        # # Reset the logging level, as Toil may have changed it
        # _logging.set_level(self.operation.log_level)
        super().teardown()


class CWLToolRunner(CWLRunner):
    """
    Wrapper class for the CWLTool CWL runner
    """
    def __init__(self, operation: Operation) -> None:
        logger.debug("CWLToolRunner.__init__()")
        super().__init__(operation)
        self.command = "cwltool"

    def setup(self) -> None:
        """
        Set arguments that are specific to this CWL runner.
        """
        logger.debug("CWLToolRunner.set_arguments()")
        super().setup()
        self.args.extend(['--disable-color'])
        self.args.extend(['--parallel'])
        self.args.extend(['--timestamps'])
        if self.operation.debug_workflow:
            self.args.extend(["--debug"])

    # def teardown(self) -> None:
    #     super().teardown()

def create_cwl_runner(runner: str, operation: Operation) -> CWLRunner:
    """
    Factory method that creates a CWLRunner instance based on the `runner` argument.
    We need access to some information inside the `operation` that calls us.
    """
    logger.debug("create_cwl_runner(%s, %s)", runner, operation)
    if runner.lower() == "toil":
        return ToilRunner(operation)
    if runner.lower() == "cwltool":
        return CWLToolRunner(operation)
    raise ValueError(f"Don't know how to create CWL runner '{runner}'")
