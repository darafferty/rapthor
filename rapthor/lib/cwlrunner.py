"""
Classes that wrap the CWL runners that Rapthor supports.
"""
from __future__ import annotations

import glob
import logging
import os
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING, Union

import yaml

if TYPE_CHECKING:
    from rapthor.lib.operation import Operation

logger = logging.getLogger("rapthor:cwlrunner")


class BaseCWLRunner:
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

    def setup(self) -> None:
        """
        Prepare runner for running. Set up the list of arguments to pass to the
        actual CWL runner.
        Derived classes can add their own arguments by overriding this method.
        """
        self.args.extend(["--outdir", self.operation.pipeline_working_dir])

        # Make a copy of the environment to allow changes made to it in the
        # setup() method of derived classes to be reverted in teardown()
        self._environment = os.environ.copy()

    def teardown(self) -> None:
        """
        Clean up after the runner has run.
        """
        os.environ.clear()
        os.environ.update(self._environment)

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

        # Ensure our custom batch system is on the path
        # So that toil can load it as a plugin
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_dir = os.path.join(script_dir, "toil_batch_systems")
        current_env = os.environ.copy()
        python_path = os.pathsep.join([current_env.get("PYTHONPATH", ""), script_dir])
        modified_env = current_env | {"PYTHONPATH": python_path}

        logger.debug("Executing command: %s", ' '.join(args))
        with open(self.operation.pipeline_outputs_file, 'w') as stdout, \
             open(self.operation.pipeline_log_file, 'w') as stderr:
            try:
                result = subprocess.run(args=args, env=modified_env, stdout=stdout, stderr=stderr, check=True)
                logger.debug(str(result))
                return True
            except subprocess.CalledProcessError as err:
                logger.critical(str(err))
                return False


class CWLRunner(BaseCWLRunner):

    def _create_mpi_config_file(self) -> None:
        """
        Create the config file for MPI jobs and add the required args
        """
        logger.debug("Creating MPI config file %s", self.operation.mpi_config_file)
        if self.operation.batch_system == 'slurm':
            mpi_config_lines = [
                "runner: 'mpi_runner.sh'",
                "nproc_flag: '-N'",
                f"extra_flags: ['--cpus-per-task={self.operation.cpus_per_task}', "
                "'mpirun', '-pernode', '--bind-to', 'none', '-x', 'OPENBLAS_NUM_THREADS']"
            ]
        else:
            mpi_config_lines = [
                "runner: 'mpirun'",
                "nproc_flag: '-np'",
                "extra_flags: ['-pernode', '--bind-to', 'none', '-x', 'OPENBLAS_NUM_THREADS']"
            ]
            if self.operation.batch_system != 'slurm_static':
                logger.warning('MPI support for non-Slurm clusters is experimental. '
                            'Please report any issues encountered.')
        with open(self.operation.mpi_config_file, 'w') as cfg_file:
            cfg_file.write('\n'.join(mpi_config_lines))

    def _delete_mpi_config_file(self) -> None:
        """
        Delete the MPI config file
        """
        os.remove(self.operation.mpi_config_file)

    def _get_tmpdir_prefix(self) -> Union[str, None]:
        """
        Return the prefix to be passed as value to the command-line option
        `--tmpdir-prefix` to the CWL runner or `None`. It is assumed that
        all CWL runners support this command-line option.

        The temporary directory is used to store intermediate results of
        a single job. It can typically be on a local (fast) scratch disk,
        unless the job uses MPI. MPI requires that intermediate results
        are accessible by all the nodes that participate in that particular
        job. Currently, only some jobs in the `Imaging` operation will use
        MPI, if the `use_mpi` option in the `[imaging]` section of the parset
        is set to `True`.

        The path to the local scratch directory can be set using the
        `local_scratch_dir` option in the `[cluster]` section of the parset.
        Similarly, the path to the shared directory can be set using the
        `global_scratch_dir` option. The `dir_local` option is now deprecated,
        but will be used for backward compatibility if `local_scratch_dir`
        is not set.

        If `global_scratch_dir` is not set when using MPI, return a
        subdirectory of the pipeline's working directory, which is guaranteed
        to be on a shared disk. Otherwise, if neither `local_scratch_dir`
        nor `dir_local` are set, return `None`.

        The file-part of the prefix is set to the name of the CWL runner.
        """
        if self.operation.use_mpi:
            prefix = (
                self.operation.global_scratch_dir
                if self.operation.global_scratch_dir
                else os.path.join(self.operation.pipeline_working_dir, "tmp")
            )
        else:
            prefix = (
                self.operation.local_scratch_dir
                if self.operation.local_scratch_dir
                else self.operation.scratch_dir
            )
        return os.path.join(prefix, self.command + ".") if prefix else None

    def _get_tmp_outdir_prefix(self) -> Union[str, None]:
        """
        Return the prefix to be passed as value to the command-line option
        `--tmp-outdir-prefix` to the CWL runner, or `None`. It is assumed
        that all CWL runners support this command-line option.

        The temporary output directory is used to store the outputs of the
        different jobs that make up a complete workflow. This directory
        typically needs to be on a global (shared) file, because the job
        results need to be visible for all the jobs in the workflow.

        The path to the global scratch directory can be set using the
        `global_scratch_dir` option in the `[cluster]` section of the
        parset. Return `None`, if `global_scratch_dir` is not set.

        The file-part of the prefix is set to the name of the CWL runner.
        """
        prefix = self.operation.global_scratch_dir
        return os.path.join(prefix, self.command + ".") if prefix else None

    def setup(self) -> None:
        """
        Prepare runner for running. Adds some additional preparations to base class.
        """
        super().setup()
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
        if self.operation.use_mpi:
            self._create_mpi_config_file()
            self.args.extend(['--mpi-config-file', self.operation.mpi_config_file])
            self.args.extend(['--enable-ext'])
        if prefix := self._get_tmpdir_prefix():
            self.args.extend(['--tmpdir-prefix', prefix])

    def teardown(self) -> None:
        """
        Clean up after the runner has run.
        """
        if self.operation.use_mpi:
            self._delete_mpi_config_file()
        super().teardown()


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

    def _get_tmp_outdir_prefix(self) -> Union[str, None]:
        """
        Return the prefix to be passed as value to the command-line option
        `--tmp-outdir-prefix` to the CWL runner or `None`. When using Slurm,
        the temporary output directory must be on a shared file system. Ensure
        this is the case by using a temporary directory inside the pipeline's
        working directory as fall-back.
        """
        prefix = super()._get_tmp_outdir_prefix()
        if not prefix and self.operation.batch_system.startswith("slurm"):
            prefix = os.path.join(
                self.operation.pipeline_working_dir, "tmp-out", self.command + "."
            )
        return prefix

    def _get_workdir(self) -> Union[str, None]:
        """
        Return the working directory for Toil, if using Slurm, else return
        `None`.  When using Slurm the working directory needs to be on a
        shared file system. Use `global_scratch_dir`, if defined in the section
        `[cluster]` of the parset file, else use a temporary directory inside
        the pipeline working directory.
        """
        if self.operation.batch_system.startswith("slurm"):
            return os.path.join(
                self.operation.global_scratch_dir
                if self.operation.global_scratch_dir
                else os.path.join(self.operation.pipeline_working_dir, "workdir"),
                ""  # adds a trailing directory separator, required by Toil
            )
        else:
            return None

    def _add_slurm_options(self) -> None:
        """
        Add options specific for running a workflow on a Slurm cluster
        """
        self.args.extend(['--disableCaching'])
        self.args.extend(['--defaultCores', str(self.operation.cpus_per_task)])
        if self.operation.mem_per_node_gb > 0:
            self.args.extend(['--defaultMemory', f'{self.operation.mem_per_node_gb}G'])
        else:
            self.args.extend(['--dont_allocate_mem'])

        # Set any Toil-specific environment variables.
        toil_env_variables = {
            "TOIL_SLURM_ARGS": "--export=ALL"
        }
        if "TOIL_SLURM_ARGS" in self._environment:
            # Add any args already set in the existing environment
            toil_env_variables["TOIL_SLURM_ARGS"] += " " + self._environment["TOIL_SLURM_ARGS"]
        os.environ.update(toil_env_variables)

    def _add_debug_options(self) -> None:
        """
        Add options specific for debugging a workflow. Debugging of workflows
        is currently only supported when running on a single machine.
        """
        if self.operation.batch_system != 'single_machine':
            raise ValueError(
                'The debug_workflow option can only be used when batch_system = "single_machine".'
            )
        self.args.extend(['--debugWorker'])  # NOTE: stdout/stderr are not redirected to the log
        self.args.extend(['--logDebug'])

    def _add_logging_options(self) -> None:
        """
        Add options specific for logging.
        """
        self.args.extend(['--writeLogs', self.operation.log_dir])
        self.args.extend(['--writeLogsFromAllJobs', 'True'])  # also keep logs of successful jobs
        self.args.extend(['--maxLogFileSize', '1gb'])  # set to large value to prevent truncation

    def setup(self) -> None:
        """
        Prepare runner for running. Adds some additional preparations to base class.
        """
        super().setup()
        # Bypass the file store; it only has benefits when using object stores like S3
        self.args.extend(['--bypass-file-store'])
        self.args.extend(['--batchSystem', self.operation.batch_system])
        self.args.extend(['--maxLocalJobs', str(self.operation.max_nodes)])
        self.args.extend(['--maxJobs', str(self.operation.max_nodes)])
        self.args.extend(['--jobStore', self.operation.jobstore])
        self.args.extend(['--stats'])  # implicitly preserves the job store for future runs
        self.args.extend(['--servicePollingInterval', '10'])
        self._add_logging_options()
        if os.path.exists(self.operation.jobstore):
            self.args.extend(['--restart'])
        if self.operation.batch_system.startswith('slurm'):
            self._add_slurm_options()
        elif self.operation.batch_system == "single_machine":
            if tmpdir_prefix := self._get_tmpdir_prefix():
                # Toil requires that temporary directory exists
                os.makedirs(os.path.dirname(tmpdir_prefix), exist_ok=True)
        if tmp_outdir_prefix := self._get_tmp_outdir_prefix():
            # Toil requires that temporary output directory exists
            os.makedirs(os.path.dirname(tmp_outdir_prefix), exist_ok=True)
            self.args.extend(['--tmp-outdir-prefix', tmp_outdir_prefix])
        if workdir := self._get_workdir():
            # Toil requires that working directory exists
            os.makedirs(workdir, exist_ok=True)
            self.args.extend(['--workDir', workdir])
        if self.operation.keep_temporary_files:
            self.args.extend(['--cleanWorkDir', 'never'])
        if self.operation.debug_workflow:
            self._add_debug_options()

    def teardown(self) -> None:
        """
        Clean up after the runner has run.
        Toil fails to properly clean up the directories it uses for temporary
        files, at least when the option `--bypass-file-store` is used. Do the
        clean up ourselves, unless the user requested to keep those files.
        TODO: This solution suffers from a race-condition. If multiple Rapthor
        runs use the same temporary directories, then one may wipe the contents
        generated by the other. One way to tackle this is make sure that the
        prefixes are unique for each Rapthor run, e.g., by adding the process
        ID to it. Another solution is to wait for the Toil developers to come
        up with a proper fix in Toil.
        """
        if not self.operation.keep_temporary_files:
            # Remove directories used for storing intermediate job results
            if prefix := self._get_tmp_outdir_prefix():
                paths = glob.glob(f"{prefix}*")
                logger.debug(
                    "Removing temporary output directories: %s", " ".join(paths)
                )
                for path in paths:
                    shutil.rmtree(path)

            if self.operation.batch_system == "single_machine":
                # Remove directories used for storing temporary data by single jobs
                if prefix := self._get_tmpdir_prefix():
                    paths = glob.glob(f"{prefix}*")
                    logger.debug(
                        "Removing temporary directories: %s", " ".join(paths)
                    )
                    for path in paths:
                        shutil.rmtree(path)
        super().teardown()


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
        if prefix := self._get_tmp_outdir_prefix():
            self.args.extend(['--tmp-outdir-prefix', prefix])
        if self.operation.debug_workflow:
            self.args.extend(["--debug"])
        if self.operation.keep_temporary_files:
            self.args.extend(["--leave-tmpdir"])

    def teardown(self) -> None:
        return super().teardown()


class StreamFlowRunner(BaseCWLRunner):
    """
    Wrapper class for the StreamFlow CWL runner
    """
    def __init__(self, operation: Operation) -> None:
        super().__init__(operation)
        self.command = "streamflow"
        self.args.extend(['run'])
        self.streamflow_file = None

    def setup(self) -> None:
        """
        Set arguments that are specific to this CWL runner.
        """
        super().setup()
        self.args.extend(["--name", self.operation.name])
        if self.operation.debug_workflow:
            self.args.extend(["--debug"])
        workflow = {
            "type": "cwl",
            "config": {
                "file": self.operation.pipeline_parset_file,
                "settings": self.operation.pipeline_inputs_file
            },
            "bindings": [
                {"step": "/", "target": {"deployment": self.operation.name}}
            ]
        }
        if self.operation.container is not None:
            if self.operation.container == 'singularity':
                workflow["config"]["docker"] = [
                    {"step": "/", "deployment": {"type": "singularity", "config": {}}}
                ]
            else:
                raise ValueError(
                    f'The `{self.operation.container}` container engine is not supported by StreamFlow'
                )
        else:
            workflow["config"]["docker"] = [
                {"step": "/", "deployment": {"type": "none", "config": {}}}
            ]
        if self.operation.batch_system == 'single_machine':
            deployment = {
                "type": "local",
                "config": {}
            }
        elif self.operation.batch_system == 'slurm':
            deployment = {
                "type": "slurm",
                "config": {
                    "maxConcurrentJobs": self.operation.max_nodes if self.operation.max_nodes > 0 else sys.maxsize,
                    "services": {
                        self.operation.name: {
                            "cpusPerTask": self.operation.cpus_per_task
                        }
                    }
                }
            }
            if self.operation.mem_per_node_gb > 0:
                deployment["config"]["services"][self.operation.name]["mem"] = f"{self.operation.mem_per_node_gb}G"
            workflow["bindings"][0]["target"]["service"] = self.operation.name
        else:
            raise ValueError(
                f'The `{self.operation.batch_system}` batch system is not supported by StreamFlow'
            )
        if self.operation.global_scratch_dir is not None:
            deployment["workdir"] = self.operation.global_scratch_dir
        if self.operation.use_mpi:
            raise ValueError(
                'The `cwltool:MPIRequirement` extension is not supported by StreamFlow'
            )
        config = {
            "version": "v1.0",
            "workflows": {
                self.operation.name: workflow
            },
            "database": {
                "type": "default",
                "config": {
                    "connection": f"{self.operation.rapthor_working_dir}/.streamflow/sqlite.db"
                }
            },
            "deployments": {
                self.operation.name: deployment
            }
        }
        self.streamflow_file = os.path.join(self.operation.pipeline_working_dir, "streamflow.yml")
        with open(self.streamflow_file, "w") as f:
            logger.debug(f"Creating StreamFlow configuration file {self.streamflow_file}:\n{yaml.safe_dump(config)}")
            yaml.safe_dump(config, f)
        self.args.extend([self.streamflow_file])

    def teardown(self) -> None:
        if self.streamflow_file is not None and not self.operation.keep_temporary_files:
            os.remove(self.streamflow_file)
        super().teardown()


def create_cwl_runner(runner: str, operation: Operation) -> CWLRunner:
    """
    Factory method that creates a CWLRunner instance based on the `runner` argument.
    We need access to some information inside the `operation` that calls us.
    """
    if runner.lower() == "toil":
        return ToilRunner(operation)
    if runner.lower() == "cwltool":
        return CWLToolRunner(operation)
    if runner.lower() == "streamflow":
        return StreamFlowRunner(operation)
    raise ValueError(f"Don't know how to create CWL runner '{runner}'")
