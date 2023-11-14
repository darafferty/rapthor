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
    Operation to mosaic sector images
    """
    def __init__(self, field, index):
        super().__init__(field, name='concatenate', index=index)

    def set_parset_parameters(self):
        """
        Define parameters needed for the CWL workflow template
        """
        if self.batch_system == 'slurm':
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
        # Divide the input observations into epochs and define the input and
        # output filenames for each epoch
        times = set([obs.starttime for obs in self.field.full_observations])
        input_filenames = []
        output_filenames = []
        for epoch, time in enumerate(times):
            input_filenames.append([obs.ms_filename for obs in self.field.full_observations
                                    if obs.starttime == time])
            output_filenames.append(f'epoch_{epoch+1}_concatenated.ms')
        self.output_filenames = output_filenames

        self.input_parms = {'input_filenames': CWLDir(input_filenames).to_json(),
                            'output_filenames': output_filenames}

    def finalize(self):
        """
        Finalize this operation
        """
        # Update the field object to use the new, concatenated MS file(s)
        self.field.ms_filenames = [os.path.join(self.pipeline_working_dir, filename)
                                   for filename in self.output_filenames]
        self.field.scan_observations()
        self.field.chunk_observations(self.field.parset['selfcal_data_fraction'])

        # Finally call finalize() in the parent class
        super().finalize()
