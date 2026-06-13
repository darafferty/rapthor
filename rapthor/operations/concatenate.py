"""
Module that holds the Concatenate class
"""

import os

from rapthor.execution.flows.concatenate import concatenate_flow, concatenate_payload_from_inputs
from rapthor.lib.operation import Operation
from rapthor.lib.records import DirectoryRecord


class Concatenate(Operation):
    """
    Operation to concatenate MS files
    """

    def __init__(self, field, index):
        super().__init__(field, index=index, name="concatenate")

    def set_parset_parameters(self):
        """
        Define parameters needed by the concatenate flow.
        """
        self.parset_parms = self.flow_parset_parameters(include_pipeline_working_dir=True)

    def set_input_parameters(self):
        """
        Define inputs passed to the concatenate flow.
        """
        # Identify which epochs need concatenation and define the input and
        # output filenames for each
        input_filenames = []
        output_filenames = []
        self.final_filenames = []  # used to store the full paths for later
        for starttime, obs_list in zip(self.field.epoch_starttimes, self.field.epoch_observations):
            if len(obs_list) > 1:
                input_filenames.append(
                    DirectoryRecord([obs.ms_filename for obs in obs_list]).to_json()
                )
                output_filename = f"epoch_{starttime}_concatenated.ms"
                output_filenames.append(output_filename)
                self.final_filenames.append(
                    os.path.join(self.pipeline_working_dir, output_filename)
                )
            else:
                self.final_filenames.append(obs_list[0].ms_filename)

        self.input_parms = {
            "input_filenames": input_filenames,
            "data_colname": self.field.data_colname,
            "output_filenames": output_filenames,
        }

    def execute_workflow(self):
        """
        Execute concatenation through the Prefect flow and return operation outputs.
        """
        payload = concatenate_payload_from_inputs(self.input_parms, self.pipeline_working_dir)
        outputs = self.run_prefect_flow(concatenate_flow, payload)
        return True, outputs

    def finalize(self):
        """
        Finalize this operation
        """
        # Update the field object to use the new, concatenated MS file(s)
        self.field.ms_filenames = self.final_filenames
        self.field.scan_observations()
        self.field.data_colname = "DATA"

        # Finally call finalize() in the parent class
        super().finalize()
