"""
Classes that wrap the CWL runners that Rapthor supports.
"""
from __future__ import annotations
import logging
import os
import shutil
import subprocess
from typing import TYPE_CHECKING
try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version

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
        Initializer
        """
        self.args = []
        self.command = None
        self.operation = operation
        self.__environment = os.environ.copy()

    def __enter__(self) -> "CWLRunner":
        """
        Called when entering a context.
        """
        self.setup()
        return self

    def __exit__(self, *_) -> None:
        """
        Called when exiting a context.
        """
        self.teardown()

    def _create_mpi_config_file(self) -> None:
        """
        Create the config file for MPI jobs and add the required args
        """
        logger.debug("Creating MPI config file %s", self.operation.mpi_config_file)
        if self.operation.batch_system == 'slurm':
            mpi_config_lines = [
                "runner: 'mpi_runner.sh'",
                "nproc_flag: '-N'",
                "extra_flags: ['mpirun', '-pernode', '--bind-to', 'none', '-x', 'OPENBLAS_NUM_THREADS']"
            ]
        else:
            mpi_config_lines = [
                "runner: 'mpirun'",
                "nproc_flag: '-np'",
                "extra_flags: ['-pernode', '--bind-to', 'none', '-x', 'OPENBLAS_NUM_THREADS']"
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
        if self.operation.scratch_dir is not None and not self.operation.use_mpi:
            # Note: --tmp-outdir-prefix is not set here, as its value depends on
            # runner/mode used
            prefix = os.path.join(self.operation.scratch_dir, self.command + '.')
            self.args.extend(['--tmpdir-prefix', prefix])
        if self.operation.use_mpi:
            self._create_mpi_config_file()
            self.args.extend(['--mpi-config-file', self.operation.mpi_config_file])
            self.args.extend(['--enable-ext'])

            # MPI requires that --tmpdir-prefix points to a shared filesystem
            # (so that all workers can access the files), so here we set it to
            # the output directory
            prefix = os.path.join(self.operation.pipeline_working_dir, self.command + '.')
            self.args.extend(['--tmpdir-prefix', prefix])

    def teardown(self) -> None:
        """
        Clean up after the runner has run.
        """
        os.environ.clear()
        os.environ.update(self.__environment)
        if self.operation.use_mpi:
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
        if self.command is None:
            raise RuntimeError(
                "Don't know how to start CWL runner {}".format(self.__class__.__name__)
            )
        args = [self.command] + self.args
        args.extend([self.operation.pipeline_parset_file,
                     self.operation.pipeline_inputs_file])
        logger.debug("Executing command: %s", ' '.join(args))
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
        super().__init__(operation)
        self.command = "toil-cwl-runner"
        self.toil_env_variables = {
            "TOIL_SLURM_ARGS": "--export=ALL"
        }
        if "TOIL_SLURM_ARGS" in self.__environment:
            # Add user-defined args
            self.toil_env_variables["TOIL_SLURM_ARGS"] += " " + self.__environment["TOIL_SLURM_ARGS"]

    def setup(self):
        """
        Prepare runner for running. Adds some additional preparations to base class.
        """
        super().setup()
        self.args.extend(['--batchSystem', self.operation.batch_system])
        # Bypass the file store; it only has benefits when using object stores like S3
        self.args.extend(['--bypass-file-store'])
        if self.operation.batch_system == 'slurm':
            self.args.extend(['--disableCaching'])
            self.args.extend(['--defaultCores', str(self.operation.cpus_per_task)])
            if self.operation.mem_per_node_gb > 0:
                self.args.extend(['--defaultMemory', f'{self.operation.mem_per_node_gb}G'])
            else:
                self.args.extend(['--dont_allocate_mem'])

            # When the slurm batch system is used, the use of --bypass-file-store
            # requires that --tmp-outdir-prefix points to a shared filesystem, so
            # here we set it to the output directory
            prefix = os.path.join(self.operation.pipeline_working_dir, self.command + '.')
            self.args.extend(['--tmp-outdir-prefix', prefix])
        self.args.extend(['--maxLocalJobs', str(self.operation.max_nodes)])
        self.args.extend(['--maxJobs', str(self.operation.max_nodes)])
        self.args.extend(['--jobStore', self.operation.jobstore])
        if os.path.exists(self.operation.jobstore):
            self.args.extend(['--restart'])
        self.args.extend(['--writeLogs', self.operation.log_dir])
        if int(version('toil').split('.')[0]) >= 6:
            # With Toil v6.0.0, the way in which several args are parsed was changed
            self.args.extend(['--writeLogsFromAllJobs', 'True'])  # also keep logs of successful jobs
            self.args.extend(['--maxLogFileSize', '1gb'])  # set to large value to prevent truncation
        else:
            self.args.extend(['--writeLogsFromAllJobs'])  # also keep logs of successful jobs
            self.args.extend(['--maxLogFileSize', '0'])  # disable truncation of log files
        if self.operation.scratch_dir is not None:
            # Note: option --workDir seems to take precedence over both --tmpdir-prefix,
            #       and --tmp-outdir-prefix. So, we may not want to set it.
            # Note: add a trailing directory separator, required by Toil v5.3+, using
            #       os.path.join()
            self.args.extend(['--workDir', os.path.join(self.operation.scratch_dir, '')])
            if self.operation.batch_system != 'slurm':
                # For non-slurm batch systems, set --tmp-outdir-prefix to the scratch
                # directory (when the slurm batch system is used, --tmp-outdir-prefix
                # is set above to a shared filesystem)
                prefix = os.path.join(self.operation.scratch_dir, self.command + '.')
                self.args.extend(['--tmp-outdir-prefix', prefix])
        if self.operation.coordination_dir is not None:
            self.args.extend(['--coordinationDir', self.operation.coordination_dir])
        self.args.extend(['--stats'])  # implicitly preserves the job store for future runs
        self.args.extend(['--servicePollingInterval', '10'])
        if self.operation.debug_workflow:
            if self.operation.batch_system != 'single_machine':
                raise ValueError(
                    'The debug_workflow option can only be used when batch_system = "single_machine".'
                )
            self.args.extend(['--cleanWorkDir', 'never'])
            self.args.extend(['--debugWorker'])  # NOTE: stdout/stderr are not redirected to the log
            self.args.extend(['--logDebug'])
        os.environ.update(self.toil_env_variables)

    def teardown(self):
        """
        Clean up after the runner has run.
        """
        super().teardown()
        if not self.operation.debug_workflow:
            # Use the logs to find the temporary directory we ran in.
            workerlogs = os.popen("grep 'Redirecting logging to' " + os.path.join(self.operation.log_dir, 'pipeline.log') + "  | awk '{print $NF}'").read()
            leftover_tempdirs = []
            for f in workerlogs.splitlines():
                tempdir = '/'.join(f.split('/')[:-2])
                if tempdir not in leftover_tempdirs:
                    leftover_tempdirs.append(tempdir)
            for t in leftover_tempdirs:
                logger.debug('Cleaning up temporary directory {:s} of {:s}'.format(t, self.operation.name))
                shutil.rmtree(t, ignore_errors=True)


class CWLToolRunner(CWLRunner):
    """
    Wrapper class for the CWLTool CWL runner
    """
    def __init__(self, operation: Operation) -> None:
        super().__init__(operation)
        self.command = "cwltool"

    def setup(self) -> None:
        """
        Set arguments that are specific to this CWL runner.
        """
        super().setup()
        self.args.extend(['--disable-color'])
        self.args.extend(['--parallel'])
        self.args.extend(['--timestamps'])
        if self.operation.scratch_dir is not None:
            prefix = os.path.join(self.operation.scratch_dir, self.command + '.')
            self.args.extend(['--tmp-outdir-prefix', prefix])
        if self.operation.debug_workflow:
            self.args.extend(["--debug"])
            self.args.extend(["--leave-tmpdir"])

    def teardown(self) -> None:
        return super().teardown()


def create_cwl_runner(runner: str, operation: Operation) -> CWLRunner:
    """
    Factory method that creates a CWLRunner instance based on the `runner` argument.
    We need access to some information inside the `operation` that calls us.
    """
    if runner.lower() == "toil":
        return ToilRunner(operation)
    if runner.lower() == "cwltool":
        return CWLToolRunner(operation)
    raise ValueError(f"Don't know how to create CWL runner '{runner}'")
