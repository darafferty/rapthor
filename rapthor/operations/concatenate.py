"""
Module that holds the Concatenate class
"""
import os
import logging
from rapthor.lib.operation import Operation
from rapthor.lib.cwl import CWLDir

log = logging.getLogger('rapthor:concatenate')


class Concatenate(Operation):
    """
    Operation to concatenate MS files
    """
    def __init__(self, field, index):
        super().__init__(field, index=index, name='concatenate')

    def set_parset_parameters(self):
        """
        Define parameters needed for the CWL workflow template
        """
        if self.batch_system.startswith('slurm'):
            # For some reason, setting coresMax ResourceRequirement hints does
            # not work with SLURM
            max_cores = None
        else:
            max_cores = self.field.parset['cluster_specific']['max_cores']
        self.parset_parms = {'rapthor_pipeline_dir': self.rapthor_pipeline_dir,
                             'pipeline_working_dir': self.pipeline_working_dir,
                             'max_cores': max_cores}

    def set_input_parameters(self):
        """
        Define the CWL workflow inputs
        """
        # Identify which epochs need concatenation and define the input and
        # output filenames for each
        input_filenames = []
        output_filenames = []
        self.final_filenames = []  # used to store the full paths for later
        for starttime, obs_list in zip(self.field.epoch_starttimes, self.field.epoch_observations):
            if len(obs_list) > 1:
                input_filenames.append(CWLDir([obs.ms_filename for obs in obs_list]).to_json())
                output_filename = f'epoch_{starttime}_concatenated.ms'
                output_filenames.append(output_filename)
                self.final_filenames.append(os.path.join(self.pipeline_working_dir, output_filename))
            else:
                self.final_filenames.append(obs_list[0].ms_filename)

        self.input_parms = {'input_filenames': input_filenames,
                            'data_colname': self.field.data_colname,
                            'output_filenames': output_filenames}

    def finalize(self):
        """
        Finalize this operation
        """
        # Update the field object to use the new, concatenated MS file(s)
        self.field.ms_filenames = self.final_filenames
        self.field.scan_observations()
        self.field.data_colname = 'DATA'

        # Finally call finalize() in the parent class
        super().finalize()
