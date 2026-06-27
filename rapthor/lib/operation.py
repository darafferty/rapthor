"""
Definition of the master Operation class
"""

import json
import logging
import os

from rapthor.lib.context import Timer
from rapthor.lib.records import NpEncoder, clean_if_file_or_directory_record, copy_record_recursive

DIR = os.path.dirname(os.path.abspath(__file__))


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

    def __init__(self, field, index=None, name: str = ""):
        self.parset = field.parset.copy()
        self.field = field
        self.index = index
        if self.index is not None:
            self.name = f"{name.lower()}_{self.index}"
        else:
            self.name = name.lower()
        self.log = logging.getLogger(f"rapthor:{self.name}")

        working_dir = self.parset["dir_working"]

        # Workflow working dir
        self.pipeline_working_dir = os.path.join(working_dir, "pipelines", self.name)
        os.makedirs(self.pipeline_working_dir, exist_ok=True)

        self.keep_temporary_files = (
            self.parset["cluster_specific"]["keep_temporary_files"]
            or self.parset["cluster_specific"]["debug_workflow"]
        )

        # Directory that holds the workflow logs in a convenient place
        log_dir = os.path.join(working_dir, "logs", self.name)
        os.makedirs(log_dir, exist_ok=True)

        # Path to preserved pipeline templates and static reference material.
        rapthor_root_dir = os.path.split(DIR)[0]
        self.rapthor_pipeline_dir = os.path.join(rapthor_root_dir, "pipeline")

        self.pipeline_inputs_file = os.path.join(self.pipeline_working_dir, "pipeline_inputs.json")

        # File indicating whether a step was completely done.
        self.done_file = os.path.join(self.pipeline_working_dir, ".done")
        self.outputs_file = os.path.join(self.pipeline_working_dir, ".outputs.json")

        # Get the batch system to use
        self.batch_system = self.parset["cluster_specific"]["batch_system"]

        self.outputs = {}

    def flow_max_cores(self):
        """
        Return the max_cores hint used by flow-backed operation payloads.

        Slurm-style execution manages cores via allocation settings, so the
        operation payload should not add a separate max_cores hint there.
        """
        if self.batch_system.startswith("slurm"):
            return None
        return self.parset["cluster_specific"]["max_cores"]

    def flow_parset_parameters(self, include_pipeline_working_dir=False, **extra):
        """
        Return common parset parameters used by flow-backed operation adapters.
        """
        parameters = {
            "rapthor_pipeline_dir": self.rapthor_pipeline_dir,
        }
        if include_pipeline_working_dir:
            parameters["pipeline_working_dir"] = self.pipeline_working_dir
        parameters["max_cores"] = self.flow_max_cores()
        parameters.update(extra)
        return parameters

    def set_parset_parameters(self):
        """
        Define parameters needed for the operation.

        The dictionary keys must match the parset parameters expected by the
        corresponding flow payload builder.

        The entries are defined in the subclasses as needed
        """
        self.parset_parms = {}

    def set_input_parameters(self):
        """
        Define parameters needed for the operation inputs.

        The dictionary keys must match the inputs expected by the corresponding
        flow payload builder.

        The entries are defined in the subclasses as needed
        """
        self.input_parms = {}

    def setup(self):
        """
        Set up this operation

        This writes the finalizer/debug input file used by the operation flow.
        """
        self.set_parset_parameters()

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
                copy_record_recursive(output_value, dest_dir, index=index, move=move)

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
                    clean_if_file_or_directory_record(output_value)

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
            "Legacy workflow execution has been retired from the production runtime; "
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
