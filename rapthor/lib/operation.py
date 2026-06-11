"""
Definition of the master Operation class
"""

import os
import logging
import json
from jinja2 import Environment, FileSystemLoader

from rapthor.lib.context import Timer
from rapthor.lib.cwl import NpEncoder, copy_cwl_recursive, clean_if_cwl_file_or_directory

DIR = os.path.dirname(os.path.abspath(__file__))
env_parset = Environment(loader=FileSystemLoader(os.path.join(DIR, "..", "pipeline", "parsets")))


class Operation(object):
    """
    Generic operation class

    An operation performs one part of the processing. It holds the operation
    settings, populates the workflow input records, and delegates execution to
    the concrete operation implementation. The field object is passed between
    operations, each of which updates it with variables needed by other, subsequent,
    operations.

    Parameters
    ----------
    field : Field object
        Field for this operation
    index : int, optional
        Index of the operation
    name : str, optional
        Name of the operation
    """

    def __init__(self, field, index=None, name: str = "", rootname: str = ""):
        self.parset = field.parset.copy()
        self.field = field
        self.rootname = rootname.lower() if rootname else name.lower()
        self.index = index
        if self.index is not None:
            self.name = f"{name.lower()}_{self.index}"
        else:
            self.name = name.lower()
        self.parset["op_name"] = name
        self.log = logging.getLogger(f"rapthor:{self.name}")
        self.force_serial_jobs = False  # force jobs to run serially
        self.use_mpi = False

        # Extra Toil env variables and Toil version
        self.toil_env_variables = {}

        # Rapthor working directory
        self.rapthor_working_dir = self.parset["dir_working"]

        # Workflow working dir
        self.pipeline_working_dir = os.path.join(self.rapthor_working_dir, "pipelines", self.name)
        os.makedirs(self.pipeline_working_dir, exist_ok=True)

        # Legacy runtime settings retained for compatibility with older parsets.
        self.cwl_runner = self.parset["cluster_specific"].get("cwl_runner")
        self.debug_workflow = self.parset["cluster_specific"]["debug_workflow"]
        self.keep_temporary_files = (
            self.parset["cluster_specific"]["keep_temporary_files"] or self.debug_workflow
        )

        # Maximum number of nodes to use
        self.max_nodes = self.parset["cluster_specific"]["max_nodes"]

        # Directory that holds the workflow logs in a convenient place
        self.log_dir = os.path.join(self.rapthor_working_dir, "logs", self.name)
        os.makedirs(self.log_dir, exist_ok=True)

        # Paths for scripts, etc. in the rapthor install directory
        self.rapthor_root_dir = os.path.split(DIR)[0]
        self.rapthor_pipeline_dir = os.path.join(self.rapthor_root_dir, "pipeline")
        self.rapthor_script_dir = os.path.join(self.rapthor_root_dir, "scripts")

        # Preserved CWL reference template names and generated input filenames.
        # Production execution now uses Prefect/Dask, but static CWL fixtures still
        # render these templates for parity checks during the migration.
        self.pipeline_parset_template = f"{self.rootname}_pipeline.cwl"
        self.subpipeline_parset_template = None
        self.pipeline_parset_file = os.path.join(self.pipeline_working_dir, "pipeline_parset.cwl")
        self.subpipeline_parset_file = os.path.join(
            self.pipeline_working_dir, "subpipeline_parset.cwl"
        )
        self.pipeline_inputs_file = os.path.join(self.pipeline_working_dir, "pipeline_inputs.json")
        self.pipeline_outputs_file = os.path.join(
            self.pipeline_working_dir, "pipeline_outputs.json"
        )
        self.pipeline_log_file = os.path.join(self.log_dir, "pipeline.log")

        # MPI configuration file
        self.mpi_config_file = os.path.join(self.pipeline_working_dir, "mpi_config.yml")

        # Toil's jobstore path
        self.jobstore = os.path.join(self.pipeline_working_dir, "jobstore")

        # File indicating whether a step was completely done.
        self.done_file = os.path.join(self.pipeline_working_dir, ".done")
        self.outputs_file = os.path.join(self.pipeline_working_dir, ".outputs.json")

        # Get the batch system to use
        self.batch_system = self.parset["cluster_specific"]["batch_system"]

        # Get the maximum number of nodes to use
        if self.force_serial_jobs or self.batch_system == "single_machine":
            self.max_nodes = 1
        else:
            self.max_nodes = self.parset["cluster_specific"]["max_nodes"]

        # Get the number of processors per task (SLRUM only). This is passed to sbatch's
        # --cpus-per-task option (see https://slurm.schedmd.com/sbatch.html). By setting
        # this value to the number of processors per node, one can ensure that each
        # task gets the entire node to itself
        self.cpus_per_task = self.parset["cluster_specific"]["cpus_per_task"]

        # Get the amount of memory in GB per node (SLRUM only).
        self.mem_per_node_gb = self.parset["cluster_specific"]["mem_per_node_gb"]

        # Set the temp directory local to each node (DEPRECATED)
        self.scratch_dir = self.parset["cluster_specific"]["dir_local"]

        # Set the local and global scratch directories
        self.local_scratch_dir = self.parset["cluster_specific"]["local_scratch_dir"]
        self.global_scratch_dir = self.parset["cluster_specific"]["global_scratch_dir"]

        # Get the container type
        if self.parset["cluster_specific"]["use_container"]:
            self.container = self.parset["cluster_specific"]["container_type"]
        else:
            self.container = None
        self.outputs = {}

    def uses_python_flow(self):
        """
        Return True when this operation executes through a Python flow.

        Operation subclasses can override this during the Prefect/Dask migration.
        The choice is intentionally owned by the operation class, not exposed as
        a user-facing mixed-backend selector.
        """
        return True

    def set_parset_parameters(self):
        """
        Define parameters needed for the operation.

        The dictionary keys must match the jinja template variables used in the
        corresponding workflow parset.

        The entries are defined in the subclasses as needed
        """
        self.parset_parms = {}

    def set_input_parameters(self):
        """
        Define parameters needed for the operation inputs.

        The dictionary keys must match the workflow inputs defined in the corresponding
        workflow parset.

        The entries are defined in the subclasses as needed
        """
        self.input_parms = {}

    def setup(self):
        """
        Set up this operation

        This writes the finalizer/debug input file and, for static reference
        tests only, can still render a preserved CWL template.
        """
        self.set_parset_parameters()
        if not self.uses_python_flow():
            # Fill the parset template and save to a file
            self.pipeline_parset_template = env_parset.get_template(self.pipeline_parset_template)
            tmp = self.pipeline_parset_template.render(self.parset_parms)
            with open(self.pipeline_parset_file, "w") as f:
                f.write(tmp)
            if self.subpipeline_parset_template is not None:
                self.subpipeline_parset_template = env_parset.get_template(
                    self.subpipeline_parset_template
                )
                tmp = self.subpipeline_parset_template.render(self.parset_parms)
                with open(self.subpipeline_parset_file, "w") as f:
                    f.write(tmp)

        # Save the workflow inputs to a file
        self.set_input_parameters()
        with open(self.pipeline_inputs_file, "w") as f:
            f.write(json.dumps(self.input_parms, cls=NpEncoder, indent=4, sort_keys=True))

    def finalize(self):
        """
        Finalize this operation.

        Create a "done" file to indicate that this operations is done.
        Specializations should be defined in the subclasses as needed.
        """
        open(self.done_file, "w").close()

    def copy_outputs_to(
        self,
        dest_dir,
        index=None,
        include=None,
        move=False,
    ):
        """
        Copy output files to a specified directory, with optional filters.

        Parameters
        ----------
        dest_dir: str
            Path of directory to which outputs will be copied
        index : int
            If an output is a list and index is specified, only the item with the specified index
            is copied (other items in the list are ignored)
        include : list or None
            List of files to include in the copy
        move : bool, optional
            If True, move files instead of copying them
        """
        for output_key, output_value in self.outputs.items():
            if include is None or output_key in include:
                copy_cwl_recursive(output_value, dest_dir, index=index, move=move)

    def clean_outputs(self, exclude=None):
        """
        Clean temporary output files, if needed.

        Parameters
        ----------
        exclude : list or None
            List of files to exclude from the cleanup
        """
        if not self.keep_temporary_files:
            for output_key, output_value in self.outputs.items():
                if exclude is None or output_key not in exclude:
                    clean_if_cwl_file_or_directory(output_value)

    def is_done(self):
        """
        Check if this operation is done, by checking if a "done" file exists.
        """
        return os.path.isfile(self.done_file)

    def store_outputs(self):
        """
        Store outputs to a JSON file.
        """
        with open(self.outputs_file, "w") as f:
            f.write(json.dumps(self.outputs, cls=NpEncoder, indent=4, sort_keys=True))

    def load_outputs(self):
        """
        Load outputs from a JSON file.
        """
        try:
            with open(self.outputs_file, "r") as f:
                self.outputs = json.load(f)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Operation {self.name} is marked done but outputs file "
                f"{self.outputs_file} is missing"
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Operation {self.name} outputs file {self.outputs_file} is not valid JSON"
            ) from exc

    def execute_workflow(self):
        """
        Execute this operation's workflow and return ``(success, outputs)``.
        """
        raise NotImplementedError(
            "CWL execution has been retired from the production runtime; "
            "operation subclasses must implement execute_workflow() with the "
            "Prefect/Dask execution path."
        )

    def run(self):
        """
        Runs the operation
        """
        # Set up workflow inputs and call the selected operation implementation
        self.setup()
        self.log.info("<-- Operation %s started", self.name)

        # Run current operation only if it hasn't run already.
        success = self.is_done()
        if not success:
            with Timer(self.log):
                success, outputs = self.execute_workflow()
                if success:
                    self.outputs = outputs
        else:
            self.log.info("Operation %s already done, skipping.", self.name)
            # Reloads outputs
            self.load_outputs()

        # Finalize
        if success:
            self.log.info("--> Operation %s completed", self.name)
            self.finalize()
            self.store_outputs()
        else:
            raise RuntimeError(f"Operation {self.name} failed due to an error")
